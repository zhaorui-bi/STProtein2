"""RNA-only prediction with both raw and spatial frozen bases exposed."""
import numpy as np
import torch
from stprotein2.graph import physical_graph
from stprotein2.models.prototypes import prototype_diagnostics
from stprotein2.training.basic import partitions_for, tensor_input


@torch.inference_mode()
def predict(model, rna, router, config, device="cpu", full_graph=False):
    """Accepts RNAView only; measured protein is not an argument or attribute."""
    model.eval()
    graph = physical_graph(rna.coordinates, config.graph_k, config.graph_radius)
    parts = None if full_graph or config.execution_mode == "full" or model.cfg.local_operator != "csnn" else partitions_for(rna, graph, router, config)
    result = model(*tensor_input(rna, graph, device),
                   partitions=None if parts is None else [p.to(device) for p in parts])
    arrays = {k: result[k].detach().cpu().numpy() for k in
              ("prediction", "base", "rna_base", "spatial_basis_delta", "state_delta", "local_delta", "local_raw")}
    arrays["spatial_alpha"] = np.asarray(float(model.spatial_alpha))
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
    diagnostics["base_definition"] = "frozen_RNA_plus_fixed_centered_spatial_basis"
    diagnostics["spatial_alpha"] = float(model.spatial_alpha)
    return arrays, diagnostics

