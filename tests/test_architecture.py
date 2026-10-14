"""Scientific contracts; nonzero correction heads expose partitioning errors."""
import copy

import numpy as np
import pytest
import torch

from stprotein2.config import ModelConfig
from stprotein2.graph import complete_halo, physical_graph, partition_graph, validate_partition
from stprotein2.model import STProtein2, center_local
from stprotein2.prototypes import EMATeacher
from stprotein2.sheaf import normalized_coefficients, cooperative_operator
from scipy.sparse.csgraph import connected_components


torch.set_num_threads(2)


def setup_model(device="cpu", checkpoint=True):
    torch.manual_seed(92)
    cfg = ModelConfig(n_genes=12, n_proteins=5, width=24, hidden=16, members=3,
                      state_prototypes=4, niche_prototypes=3, prototype_dim=16,
                      experts=2, stalk=4, channels=4, protein_embedding=6,
                      dropout=0, checkpoint_blocks=checkpoint)
    model = STProtein2(cfg).to(device)
    # Production heads start at zero. A zero-only test would miss graph bugs.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("readout.weight") or name.endswith("readout.2.weight"):
                p.normal_(std=0.15)
    coords = np.array([(r, c) for r in range(5) for c in range(6)], dtype=float)
    graph = physical_graph(coords, k=4, radius=1.5)
    parts = partition_graph(graph, core_size=5, min_core=3, max_core=7)
    x = torch.randn(len(coords), 12, device=device)
    m = torch.ones_like(x)
    q = torch.randn(len(coords), 4, device=device)
    return model, (x, m, q, graph.torch(device)), parts


def test_halo_ownership_and_truncation():
    _, args, _ = setup_model()
    graph = physical_graph(np.array([(i, 0) for i in range(11)]), k=2, radius=1.1)
    parts = partition_graph(graph, core_size=3, min_core=2, max_core=4)
    assert validate_partition(graph, parts)
    with pytest.raises(ValueError, match="exactly one"):
        validate_partition(graph, parts + parts[:1])
    bad = [complete_halo(graph, p.core, hops=0) for p in parts]
    with pytest.raises(ValueError, match="Truncated"):
        validate_partition(graph, bad)


def test_small_core_merges_preserve_connectivity_and_budget():
    coords = np.array([(i, j) for i in range(14) for j in range(13)] + [(99, 99)], dtype=float)
    graph = physical_graph(coords, k=4, radius=1.1)
    rng = np.random.default_rng(22)
    embedding = rng.normal(size=(len(coords), 3))
    kwargs = dict(core_size=25, min_core=18, max_core=40, embedding=embedding)
    unmerged = partition_graph(graph, **kwargs, merge_small=False)
    merged = partition_graph(graph, **kwargs)
    assert len(merged) < len(unmerged)
    assert validate_partition(graph, merged)
    for part in merged:
        assert len(part.core) <= 40
        assert connected_components(graph.adjacency[part.core][:, part.core])[0] == 1
    assert any(np.array_equal(p.core, [len(coords)-1]) for p in merged)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_complete_halo_matches_full_predictions_and_gradients(device):
    model, args, parts = setup_model(device)
    model.train()
    target = torch.randn(len(args[0]), 5, device=device)
    full = model(*args)
    assert full["local_delta"].abs().max() > 1e-4
    loss = (full["prediction"] - target).square().mean()
    loss.backward()
    expected = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    tiled = model(*args, partitions=[p.to(device) for p in reversed(parts)])
    torch.testing.assert_close(tiled["prediction"], full["prediction"], atol=2e-6, rtol=2e-5)
    (tiled["prediction"] - target).square().mean().backward()
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], atol=5e-6, rtol=2e-4, msg=lambda m: name + ": " + m)


def test_isolated_and_both_gates_off():
    model, args, _ = setup_model()
    model.eval()
    result = model(*args)
    torch.testing.assert_close(result["local_delta"].sum(0), torch.zeros(5), atol=1e-6, rtol=0)
    model.set_gates(torch.zeros(5), torch.zeros(5))
    result = model(*args)
    assert torch.equal(result["prediction"], result["base"])
    graph = physical_graph(np.array([[0., 0.]]))
    isolated = model(args[0][:1], args[1][:1], args[2][:1], graph.torch())
    assert torch.count_nonzero(isolated["local_delta"]) == 0


def test_unavailable_values_cannot_affect_prediction():
    model, args, _ = setup_model()
    model.eval()
    x, m, q, graph = args
    m[:, 2] = 0
    changed = x.clone(); changed[:, 2] = 10000
    a = model(x, m, q, graph)["prediction"]
    b = model(changed, m, q, graph)["prediction"]
    assert torch.equal(a, b)


def test_no_qc_encoder_control_keeps_direct_path_invariant():
    from dataclasses import replace
    model,args,_=setup_model()
    cfg=replace(model.cfg,use_encoder_qc=False,use_state=False,use_niche=False,local_operator='none')
    direct=STProtein2(cfg).eval()
    x,m,q,graph=args
    first=direct(x,m,q,graph)['prediction']
    second=direct(x,m,q*100+20,graph)['prediction']
    assert torch.equal(first,second)


def test_teacher_stop_gradient_and_exact_ema():
    model, args, _ = setup_model()
    teacher = EMATeacher(model.representations)
    teacher.train()
    assert not teacher.training
    output = teacher(*args)
    assert not output["h"].requires_grad
    first = next(teacher.representations.parameters()).clone()
    student_p = next(model.representations.parameters())
    with torch.no_grad():
        student_p.add_(1)
    teacher.update(model.representations, 0.9)
    torch.testing.assert_close(next(teacher.representations.parameters()), first + 0.1)
    assert teacher.update_count.item() == 1
    before = teacher.state_center.clone()
    student = model.representations(*args)
    loss = teacher.loss(student, output, 0.07)
    (loss["state"] + loss["niche"]).backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert torch.equal(teacher.state_center, before)
    teacher.update_centers(output)
    torch.testing.assert_close(teacher.state_center, 0.1 * output["state"]["logits"].mean(0, keepdim=True))


def test_dense_two_operator_reference_on_directed_graph():
    # Dense matrix reference is independent of scatter-based propagation.
    torch.manual_seed(31)
    n, d, c = 6, 3, 2
    edge = torch.tensor([[0, 0, 1, 2, 3, 3, 4], [1, 2, 2, 0, 4, 5, 1]])
    t, _ = torch.linalg.qr(torch.randn(n, d, d, dtype=torch.float64))
    s, _ = torch.linalg.qr(torch.randn(n, d, d, dtype=torch.float64))
    ct = torch.rand(n, dtype=torch.float64) + 0.2
    cs = torch.rand(n, dtype=torch.float64) + 0.2
    coeff = normalized_coefficients(t, s, ct, cs, edge, n)
    x = torch.randn(n, d, c, dtype=torch.float64, requires_grad=True)
    matrices = []
    for tm, sm, diag in [(coeff.t_out, coeff.s_out, coeff.diag_out), (coeff.t_in, coeff.s_in, coeff.diag_in)]:
        matrix = torch.zeros(n*d, n*d, dtype=torch.float64)
        for i in range(n):
            matrix[i*d:(i+1)*d, i*d:(i+1)*d] += diag[i] * torch.eye(d, dtype=torch.float64)
        for i, j in edge.T.tolist():
            matrix[i*d:(i+1)*d, j*d:(j+1)*d] -= tm[i].T @ sm[j]
        matrices.append(matrix)
    expected = (matrices[1] @ matrices[0] @ x.flatten(0, 1)).reshape(n, d, c)
    actual = cooperative_operator(x, edge, coeff.tensors())
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_centering_gradient_includes_global_mean():
    raw = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    degree = torch.tensor([0, 1, 2, 1, 2, 0, 1, 1, 3])
    centered = center_local(raw, degree)
    weights = torch.randn_like(centered)
    gradient = torch.autograd.grad((centered * weights).sum(), raw)[0]
    eligible = degree > 0
    expected = torch.zeros_like(raw)
    expected[eligible] = weights[eligible] - weights[eligible].mean(0)
    torch.testing.assert_close(gradient, expected)


def test_inference_does_not_update_model_or_codebooks():
    model, args, parts = setup_model()
    model.eval()
    before = copy.deepcopy(model.state_dict())
    with torch.inference_mode():
        model(*args, partitions=[p.to("cpu") for p in parts])
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())

