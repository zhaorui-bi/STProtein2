"""Add the frozen linear spatial basis without altering neural representations."""
import math

import torch

from stprotein2.models.residual import center_local
from stprotein2.models.anchor import AnchoredSTProtein2


def spatial_basis(base, graph, alpha):
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Fixed spatial alpha must be finite and in [0,1]')
    if base.ndim != 2 or len(base) != graph.n or not base.is_floating_point() or not torch.isfinite(base).all():
        raise ValueError('Finite node-by-protein base required')
    if alpha == 0:
        return base, torch.zeros_like(base)
    value = base.double()
    delta = center_local(graph.mean(value) - value, graph.degree)
    result = (value + alpha * delta).to(base.dtype)
    return result, delta


class SpatialAnchoredSTProtein2(AnchoredSTProtein2):
    def __init__(self, cfg, spatial_alpha=0.5):
        if not math.isfinite(spatial_alpha) or not 0 <= spatial_alpha <= 1:
            raise ValueError('Fixed spatial alpha must be finite and in [0,1]')
        super().__init__(cfg)
        self.register_buffer('spatial_alpha', torch.tensor(float(spatial_alpha), dtype=torch.float64))

    def forward(self, x, availability, qc, graph, partitions=None, enable_corrections=True):
        result = super().forward(x, availability, qc, graph, partitions, enable_corrections)
        raw = result['base']
        base, delta = spatial_basis(raw, graph, self.spatial_alpha)
        correction = self.gamma_state * result['state_delta'] + self.gamma_local * result['local_delta']
        prediction = base + correction
        result.update(rna_base=raw, spatial_basis_delta=delta, base=base, prediction=prediction,
                      base_members=base[:, None, :], members=prediction[:, None, :])
        return result

    def architecture(self):
        result = super().architecture()
        result.update(variant='frozen_spatial_ridge_anchor', spatial_alpha=float(self.spatial_alpha),
                      spatial_alpha_trainable=False,
                      closed_gates_equal='frozen RNA anchor plus fixed centered spatial basis',
                      complete_halo_depth=2, full_slice_spatial_basis=True)
        return result
