"""Shared tensor, initialization and low-level neural prediction helpers."""

import numpy as np
import torch

from stprotein2.graph import physical_graph, partition_graph, boundary_score
from stprotein2.models.prototypes import prototype_diagnostics


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_input(view, graph, device):
    return (torch.as_tensor(view.x, device=device), torch.as_tensor(view.availability, device=device),
            torch.as_tensor(view.qc, device=device), graph.torch(device))


def partitions_for(view, graph, router, config):
    z, assignment = router.transform(view.x)
    score = boundary_score(graph, assignment)
    return partition_graph(graph, config.core_size, score, z, config.min_core, config.max_core, hops=2,
                           node_keys=view.barcodes)


@torch.inference_mode()
def predict(model, rna, router, config, device="cpu", full_graph=False):
    """Accepts RNAView only; measured protein is not an argument or attribute."""
    model.eval()
    graph = physical_graph(rna.coordinates, config.graph_k, config.graph_radius)
    parts = None if full_graph or config.execution_mode == "full" or model.cfg.local_operator != "csnn" else partitions_for(rna, graph, router, config)
    result = model(*tensor_input(rna, graph, device),
                   partitions=None if parts is None else [p.to(device) for p in parts])
    arrays = {k: result[k].detach().cpu().numpy() for k in
              ("prediction", "base", "state_delta", "local_delta", "local_raw")}
    arrays["barcodes"] = rna.barcodes
    arrays["gamma_state"] = model.gamma_state.cpu().numpy().copy()
    arrays["gamma_local"] = model.gamma_local.cpu().numpy().copy()
    arrays["core_id"] = np.zeros(len(rna.x), dtype=np.int64)
    if parts is not None:
        for i, part in enumerate(parts):
            arrays["core_id"][part.core] = i
    diagnostics = {name: prototype_diagnostics(result["representations"][name], getattr(model.representations, name))
                   for name in ("state", "niche") if result["representations"][name] is not None}
    diagnostics["partition"] = {"cores": len(parts) if parts else 1,
                                "halo_amplification": sum(len(p.nodes) for p in parts) / len(rna.x) if parts else 1,
                                "max_nodes": max(len(p.nodes) for p in parts) if parts else len(rna.x),
                                "core_sizes": [len(p.core) for p in parts] if parts else [len(rna.x)],
                                "halo_target_1024_exceeded": any(len(p.nodes) > 1024 for p in parts) if parts else False}
    return arrays, diagnostics


@torch.no_grad()
def initialize_prototypes(model, references, graphs, device, seed):
    model.eval()
    for key in ("state", "niche"):
        bank = getattr(model.representations, key)
        if bank is None:
            continue
        by_group = {}
        for sample in references:
            rep = model.representations(*tensor_input(sample.rna, graphs[sample.rna.name], device))
            by_group.setdefault(sample.rna.group, []).append(rep[key]["z"].detach().cpu())
        generator = torch.Generator().manual_seed(seed)
        # Equal group sample counts, without oversampling a small group.
        joined = {k: torch.cat(v) for k, v in by_group.items()}
        count = min(2000, min(len(z) for z in joined.values()))
        pooled = torch.cat([z[torch.randperm(len(z), generator=generator)[:count]] for z in joined.values()])
        bank.initialize(pooled, seed)
