"""Preserve an independently fitted Ridge while learning optional corrections."""
import torch
from torch import nn
from torch.nn import functional as F

from stprotein2.model import STProtein2


class FrozenRidge(nn.Module):
    """Buffers, never optimizer parameters; missing standardized inputs are zero."""

    def __init__(self, n_genes, n_proteins):
        super().__init__()
        self.register_buffer("weight", torch.zeros(n_proteins, n_genes))
        self.register_buffer("bias", torch.zeros(n_proteins))

    @torch.no_grad()
    def set_coefficients(self, coefficient, intercept):
        for name, source in (("weight", coefficient), ("bias", intercept)):
            target = getattr(self, name)
            value = torch.as_tensor(source, device=target.device, dtype=target.dtype)
            if target.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError("Invalid frozen Ridge coefficients")
            target.copy_(value)

    def forward(self, x, availability):
        available = torch.as_tensor(availability, device=x.device)
        if available.shape not in {(x.shape[1],), x.shape} or not torch.all((available == 0) | (available == 1)):
            raise ValueError("Invalid anchor availability")
        return F.linear(torch.where(available > 0, x, 0), self.weight, self.bias)


class AnchoredSTProtein2(STProtein2):
    """Student direct head is auxiliary; only prototype/local residuals deploy.

    This reuses the audited representation and spatial graph machinery, but
    does not replace the independent anchor with the jointly trained head.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.anchor = FrozenRidge(cfg.n_genes, cfg.n_proteins)

    def forward(self, x, availability, qc, graph, partitions=None, enable_corrections=True):
        result = super().forward(x, availability, qc, graph, partitions, enable_corrections)
        result["auxiliary"] = result["base"]
        result["auxiliary_members"] = result["base_members"]
        base = self.anchor(x, availability)
        correction = self.gamma_state * result["state_delta"] + self.gamma_local * result["local_delta"]
        # Do not average replicated identical predictions: round-off could
        # otherwise break the exact two-gate fallback contract.
        prediction = base + correction
        result.update(base=base, prediction=prediction,
                      base_members=base[:, None, :], members=prediction[:, None, :])
        return result

    def architecture(self):
        result = super().architecture()
        result.update(variant="frozen_ridge_anchor", anchor_parameters=0,
                      anchor_buffer_values=self.anchor.weight.numel()+self.anchor.bias.numel(),
                      auxiliary_direct_head_deployed=False,
                      closed_gates_equal="independently fitted frozen Ridge in this package's inference arithmetic")
        return result
