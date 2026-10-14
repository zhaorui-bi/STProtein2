"""Spatial bases, nested target identity, exact controls and actual recovery."""
import argparse
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from stprotein2.config import from_dict
from stprotein2.data import Cohort, PANEL
from stprotein2.graph import FrozenRouter, physical_graph
from stprotein2.cli import score_run
from stprotein2.trainer import tensor_input
from stprotein2_anchor.model import AnchoredSTProtein2
from stprotein2_crossfit import trainer as old_trainer, targets as old_targets
from stprotein2_smoothing.core import spatial_residual, apply
from stprotein2_partition import Budget
from stprotein2_spatial_anchor.config import SpatialAnchorTrainConfig
from stprotein2_spatial_anchor.model import SpatialAnchoredSTProtein2, spatial_basis
from stprotein2_spatial_anchor.prediction import predict
from stprotein2_spatial_anchor.partition import plan_for_model
from stprotein2_spatial_anchor import targets, trainer, nested
from test_crossfit import cohort_files, configs


def configuration(alpha=.5):
    model, old = configs()
    return model, SpatialAnchorTrainConfig(**old.to_dict(), spatial_alpha=alpha)


def test_basis_matches_independent_numpy_and_exact_closed_gate_fallback():
    cfg, train = configuration()
    cfg = replace(cfg, dropout=0)
    torch.manual_seed(12)
    model = SpatialAnchoredSTProtein2(cfg, .5)
    model.anchor.set_coefficients(np.random.default_rng(1).normal(size=(31, 6)), np.zeros(31))
    coordinates = np.array([(i // 4, i % 4) for i in range(12)] + [(100, 100)], dtype=float)
    graph = physical_graph(coordinates, k=4, radius=1.1)
    x = torch.randn(len(coordinates), 6); availability = torch.ones(6); qc = torch.randn(len(x), 4)
    with torch.no_grad():
        model.state_readout[-1].weight.normal_(0, .1)
        model.niche_readout[-1].weight.normal_(0, .1)
        model.expert_readout.weight.normal_(0, .1)
    model.eval(); model.set_gates(np.zeros(31), np.zeros(31))
    result = model(x, availability, qc, graph.torch())
    raw = result['rna_base'].detach().numpy()
    expected = apply(raw, spatial_residual(raw, graph), .5)
    np.testing.assert_array_equal(result['base'].detach().numpy(), expected)
    assert torch.equal(result['prediction'], result['base'])
    assert torch.equal(result['base'][-1], result['rna_base'][-1])
    assert torch.count_nonzero(result['spatial_basis_delta'][-1]) == 0
    torch.testing.assert_close(result['spatial_basis_delta'].sum(0), torch.zeros(31, dtype=torch.float64), rtol=0, atol=1e-12)
    original, zeros = spatial_basis(result['rna_base'], graph.torch(), 0.)
    assert original is result['rna_base'] and torch.count_nonzero(zeros) == 0
    assert 'spatial_alpha' not in dict(model.named_parameters())
    with pytest.raises(ValueError, match='alpha'):
        replace(train, spatial_alpha=float('nan'))


def test_alpha_zero_matches_native_neural_outputs_and_every_parameter_gradient():
    cfg, _ = configuration(0.)
    cfg = replace(cfg, dropout=0)
    torch.manual_seed(17); before = AnchoredSTProtein2(cfg)
    torch.manual_seed(17); after = SpatialAnchoredSTProtein2(cfg, 0.)
    with torch.no_grad():
        for name, p in before.named_parameters():
            if 'readout' in name: p.normal_(0, .1)
        after.load_state_dict({**before.state_dict(), 'spatial_alpha': after.spatial_alpha})
    graph = physical_graph(np.array([(i // 4, i % 4) for i in range(12)], dtype=float))
    args = (torch.randn(12, 6), torch.ones(6), torch.randn(12, 4), graph.torch())
    y = torch.randn(12, 31)
    a, b = before(*args), after(*args)
    assert a['local_delta'].abs().max() > 1e-6
    for key in ('prediction', 'base', 'state_delta', 'local_delta', 'auxiliary_members'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    losses_a = old_trainer.supervised_losses(a, y, a['base'].detach(), before.gamma_state, before.gamma_local)
    losses_b = trainer.supervised_losses(b, y, b['base'].detach(), after.gamma_state, after.gamma_local)
    (losses_a[0] + .2 * losses_a[1]).backward(); (losses_b[0] + .2 * losses_b[1]).backward()
    for name, p in before.named_parameters():
        q = dict(after.named_parameters())[name]
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_spatial_basis_nonzero_full_partition_predictions_and_gradients():
    cfg, _ = configuration(); cfg = replace(cfg, dropout=0)
    model = SpatialAnchoredSTProtein2(cfg)
    with torch.no_grad():
        model.anchor.weight.normal_()
        model.state_readout[-1].weight.normal_(0, .1)
        model.niche_readout[-1].weight.normal_(0, .1)
        model.expert_readout.weight.normal_(0, .1)
    graph = physical_graph(np.array([(i // 6, i % 6) for i in range(30)], dtype=float), k=4, radius=1.5)
    plan = plan_for_model(model, graph, method='grid', target_core=6, min_core=3, max_core=8,
        budget=Budget(max_nodes=20), refinement={'max_evaluations': 100})
    args = (torch.randn(30, 6), torch.ones(6), torch.randn(30, 4), graph.torch())
    target = torch.randn(30, 31)
    full = model(*args)
    assert full['local_delta'].abs().max() > 1e-6 and full['spatial_basis_delta'].abs().max() > 1e-6
    F.mse_loss(full['prediction'], target).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    split = model(*args, partitions=[p.to('cpu') for p in reversed(plan.parts)])
    torch.testing.assert_close(split['prediction'], full['prediction'], rtol=2e-5, atol=2e-6)
    F.mse_loss(split['prediction'], target).backward()
    for name, p in model.named_parameters():
        if name in grads: torch.testing.assert_close(p.grad, grads[name], rtol=2e-4, atol=5e-6)
    with pytest.raises(ValueError, match='overridden'):
        plan_for_model(model, graph, required_hops=0)


def test_oof_spatial_target_uses_only_excluded_source_anchor_and_matching_graph(tmp_path):
    cohort, _, groups = cohort_files(tmp_path); cfg, train = configuration()
    names = ['s0', 's1', 's2', 's3']
    full = targets.get_anchor(cohort, names, 6)
    baseline = targets.build_targets(cohort, names, 6, train, full)
    refs = cohort.references(names, full[0]['transform'])
    targets.validate_targets(refs, baseline, train)
    for name, value in baseline.items():
        lineage = value.provenance
        assert groups[name] not in lineage['anchor_reference_groups']
        assert not set(lineage['anchor_reference_names']) & {'s4', 's5', 's6'}
        for fold in lineage['anchor_cv']:
            assert all(groups[n] != groups[name] for n in fold['training'] + fold['validation'])
        view = next(s.rna for s in refs if s.rna.name == name)
        graph = physical_graph(view.coordinates, train.graph_k, train.graph_radius)
        expected = apply(value.rna_prediction, spatial_residual(value.rna_prediction, graph), train.spatial_alpha)
        np.testing.assert_array_equal(expected, value.prediction)
    assert not set(cohort.protein_reads) & {'s4', 's5', 's6'}
    bad = dict(baseline); bad['s0'] = replace(baseline['s0'], prediction=baseline['s0'].prediction+.1)
    with pytest.raises(ValueError, match='frozen RNA base'):
        targets.validate_targets(refs, bad, train)
    with pytest.raises(ValueError, match='graph, alpha'):
        targets.validate_targets(refs, baseline, replace(train, spatial_alpha=.75))


def test_alpha_zero_actual_training_update_matches_frozen_crossfit(tmp_path):
    cohort, _, _ = cohort_files(tmp_path); cfg, train = configuration(0.)
    train = replace(train, epochs=2)
    _, old_cfg = configs(); old_cfg = replace(old_cfg, epochs=2)
    names = ['s0', 's1', 's2']; full = targets.get_anchor(cohort, names, 6)
    new_targets = targets.build_targets(cohort, names, 6, train, full)
    old_baselines = old_targets.build_targets(cohort, names, 6, old_cfg.residual_target, full)
    refs = cohort.references(names, full[0]['transform'])
    router = FrozenRouter(components=3, clusters=3).fit([s.rna.x for s in refs], [s.rna.group for s in refs])
    old, a = old_trainer.fit(refs, cfg, old_cfg, router, tmp_path/'old', full[0]['model'], old_baselines)
    new, b = trainer.fit(refs, cfg, train, router, tmp_path/'new', full[0]['model'], new_targets)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    assert [r['loss'] for r in a['history']] == [r['loss'] for r in b['history']]
    assert [r['teacher_updates'] for r in a['history']] == [r['teacher_updates'] for r in b['history']]


def test_spatial_training_resume_uses_spatial_validation_and_rejects_changed_coefficient(tmp_path, monkeypatch):
    cohort, _, _ = cohort_files(tmp_path); cfg, train = configuration()
    names = ['s0', 's1', 's2']; full = targets.get_anchor(cohort, names, 6)
    baselines = targets.build_targets(cohort, names, 6, train, full)
    refs = cohort.references(names, full[0]['transform']); val = cohort.references(['s4'], full[0]['transform'])
    router = FrozenRouter(components=3, clusters=3).fit([s.rna.x for s in refs], [s.rna.group for s in refs])
    expected, a = trainer.fit(refs, cfg, train, router, tmp_path/'continuous', full[0]['model'], baselines, validation=val)
    save = trainer.save_torch
    def interrupt(path, value):
        save(path, value)
        if path.name == 'resume.pt' and value['epoch'] == 1: raise RuntimeError('injected interruption')
    monkeypatch.setattr(trainer, 'save_torch', interrupt)
    with pytest.raises(RuntimeError, match='injected'):
        trainer.fit(refs, cfg, train, router, tmp_path/'resumed', full[0]['model'], baselines, validation=val)
    monkeypatch.setattr(trainer, 'save_torch', save)
    actual, b = trainer.fit(refs, cfg, train, router, tmp_path/'resumed', full[0]['model'], baselines, validation=val)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
    output, _ = predict(actual, val[0].rna, router, train)
    mse = float(np.mean((output['prediction']-val[0].target)**2, dtype=np.float64))
    assert mse == b['reference_validation_mse'] and b['anchor_unchanged']
    assert b['validation_baseline'] == 'full_allowed_training_anchor_plus_fixed_spatial_basis'
    loaded = nested.load_model(tmp_path/'resumed/model.pt', 'cpu')
    assert float(loaded.spatial_alpha) == .5
    loaded.set_gates(np.zeros(31), np.zeros(31))
    closed, _ = predict(loaded, val[0].rna, router, train)
    np.testing.assert_array_equal(closed['prediction'], closed['base'])
    changed = replace(train, spatial_alpha=.75)
    new_bases = targets.build_targets(cohort, names, 6, changed, full)
    with pytest.raises(RuntimeError, match='changed'):
        trainer.fit(refs, cfg, changed, router, tmp_path/'resumed', full[0]['model'], new_bases, validation=val)


def test_full_nested_spatial_protocol_has_no_query_or_gate_source_in_oof_anchor(tmp_path, monkeypatch):
    _, registry, groups = cohort_files(tmp_path); cfg, train = configuration(); train = replace(train, epochs=2)
    mapping = tmp_path/'groups.json'; mapping.write_text(json.dumps({'groups': groups}))
    spec = tmp_path/'config.json'; spec.write_text(json.dumps({'model': cfg.to_dict(), 'train': train.to_dict()}))
    output = tmp_path/'nested'
    args = argparse.Namespace(registry=str(registry), groups=str(mapping), config=str(spec), query='s6',
        output=str(output), anchor_cache=str(tmp_path/'anchors'), device='cpu', reuse_anchor_cache=[])
    reader = Cohort.measured_protein
    def guard(self, name):
        assert name != 's6'
        return reader(self, name)
    monkeypatch.setattr(Cohort, 'measured_protein', guard)
    nested.run_nested(args)
    reports = list(output.rglob('training.json')); assert len(reports) == 31
    for path in reports:
        row = json.loads(path.read_text())
        assert row['selected_teacher_updates'] > 0 and row['anchor_unchanged']
        assert 's6' not in row['reference_names'] + row['inner_validation']
        for target in json.loads((path.parent/'residual_baselines.json').read_text()).values():
            lineage = target['provenance']
            assert lineage['spatial_basis']['alpha'] == .5
            assert set(lineage['anchor_reference_names']) <= set(row['reference_names'])
            assert not set(lineage['anchor_reference_names']) & set(row['inner_validation'])
    for path in output.glob('gate_oof_*/component_losses.json'):
        held = json.loads(path.read_text())['held_group']
        for record in path.parent.rglob('residual_baselines.json'):
            for target in json.loads(record.read_text()).values():
                assert held not in target['provenance']['anchor_reference_groups']
    assert not (output/'development_scores.json').exists()
    saved = np.load(output/'predictions/s6/prediction.npz')['prediction'].copy()
    nested.run_nested(args)
    (output/'prediction_complete.json').unlink()
    nested.run_nested(args)
    np.testing.assert_array_equal(saved, np.load(output/'predictions/s6/prediction.npz')['prediction'])
    monkeypatch.setattr(Cohort, 'measured_protein', reader)
    score_run(args)
    with (output/'model.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(RuntimeError, match='deployment artifact changed'):
        nested.run_nested(args)
