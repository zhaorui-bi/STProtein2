"""Optional compact RNA residuals; no prototype, sheaf, Teacher or routing code.

The layer algebra and ordinary residuals preserve their registered research
implementations. Conversion keeps trained tensors and removes auxiliary-only
parameters; it does not fit or select a model.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResidualConfig:
    n_genes: int = 4000
    n_proteins: int = 31
    width: int = 256
    hidden: int = 128
    members: int = 8
    qc_dim: int = 4
    dropout: float = .1
    use_encoder_qc: bool = True
    residual_head: str = 'own_mlp'
    residual_width: int = 256
    weight_modulated_input: bool = True

    def __post_init__(self):
        for key in ('n_genes', 'n_proteins', 'width', 'hidden', 'members', 'qc_dim', 'residual_width'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Positive integer required: ' + key)
        if self.n_proteins != 31 or self.qc_dim != 4:
            raise ValueError('The fixed 31-channel and four-QC contract is required')
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError('dropout must be finite in [0,1)')
        if self.residual_head not in {'own_mlp', 'neighbor_mlp'}:
            raise ValueError('Only the registered compact residual heads are supported')
        if not isinstance(self.use_encoder_qc, bool) or not isinstance(self.weight_modulated_input, bool):
            raise ValueError('Execution and QC flags must be boolean')


class EnsembleLinear(nn.Module):
    def __init__(self, n_in, n_out, members, weight_modulated_input=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.r = nn.Parameter(torch.empty(members, n_in))
        self.s = nn.Parameter(torch.empty(members, n_out))
        self.bias = nn.Parameter(torch.zeros(members, n_out))
        self.weight_modulated_input = weight_modulated_input
        nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
        with torch.no_grad():
            self.r.bernoulli_(.5).mul_(2).sub_(1)
            self.s.fill_(1)

    def forward(self, x):
        if x.ndim == 2 and self.weight_modulated_input:
            members, genes = self.r.shape
            output = self.weight.shape[1]
            weights = (self.weight[:, None, :] * self.r.T[:, :, None]).reshape(genes, members * output)
            value = (x @ weights).reshape(len(x), members, output)
            return value * self.s + self.bias
        if x.ndim == 2:
            x = x[:, None, :]
        return ((x * self.r) @ self.weight) * self.s + self.bias


class RNAEnsemble(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            EnsembleLinear(cfg.n_genes, cfg.width, cfg.members, cfg.weight_modulated_input),
            EnsembleLinear(cfg.width, cfg.hidden, cfg.members, cfg.weight_modulated_input)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.hidden)])
        self.mask = nn.Sequential(nn.Linear(cfg.n_genes, 32), nn.GELU(), nn.Linear(32, cfg.hidden))
        self.qc = nn.Linear(cfg.qc_dim, cfg.hidden)
        self.output_norm = nn.LayerNorm(cfg.hidden)
        self.dropout, self.use_encoder_qc = cfg.dropout, cfg.use_encoder_qc

    def forward(self, x, availability, qc):
        if availability.ndim == 1:
            availability = availability[None, :].expand_as(x)
        if availability.shape != x.shape:
            raise ValueError('Availability must match input genes')
        h = torch.where(availability > 0, x, torch.zeros_like(x))
        for linear, norm in zip(self.layers, self.norms):
            h = F.dropout(F.gelu(norm(linear(h))), p=self.dropout, training=self.training)
        mask_h = self.mask(1 - availability)
        encoder_qc = qc if self.use_encoder_qc else torch.zeros_like(qc)
        return self.output_norm(h + mask_h[:, None, :] + self.qc(encoder_qc)[:, None, :])


def neighbor_mean(h, edge_index, degree):
    row, col = edge_index
    result = torch.zeros_like(h).index_add(0, row, h[col])
    result = result / degree.clamp_min(1).reshape(len(h), 1)
    return torch.where((degree > 0).reshape(len(h), 1), result, h)


def center_local(raw, degree):
    valid = (degree > 0).to(raw)
    mean = (raw * valid[:, None]).sum(0) / valid.sum().clamp_min(1)
    return valid[:, None] * (raw - mean)


class CompactResidual(nn.Module):
    def __init__(self, cfg=ResidualConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = RNAEnsemble(cfg)
        self._initialize_residual()

    def _initialize_residual(self):
        cfg = self.cfg
        n_in = cfg.hidden if cfg.residual_head == 'own_mlp' else 2 * cfg.hidden + cfg.qc_dim
        self.readout = nn.Sequential(nn.Linear(n_in, cfg.residual_width), nn.GELU(),
                                     nn.Linear(cfg.residual_width, cfg.n_proteins, bias=False))
        self.gate = nn.Linear(cfg.hidden + cfg.qc_dim, cfg.n_proteins)
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.constant_(self.gate.bias, -2)
        own = cfg.residual_head == 'own_mlp'
        self.register_buffer('gamma_state', torch.ones(31) if own else torch.zeros(31))
        self.register_buffer('gamma_local', torch.zeros(31) if own else torch.ones(31))

    def _check_inputs(self, x, availability, qc):
        if (x.ndim != 2 or x.shape[1] != self.cfg.n_genes or qc.shape != (len(x), self.cfg.qc_dim)
                or not torch.isfinite(x).all() or not torch.isfinite(qc).all()
                or availability.shape not in {(self.cfg.n_genes,), x.shape}
                or not torch.all((availability == 0) | (availability == 1))):
            raise ValueError('Finite aligned RNA/QC and binary availability are required')

    def forward(self, x, availability, qc, edge_index=None, degree=None):
        self._check_inputs(x, availability, qc)
        h = self.encoder(x, availability, qc).mean(1)
        return self._components(h, qc, edge_index, degree)

    def _components(self, h, qc, edge_index=None, degree=None, enabled=True):
        own = self.cfg.residual_head == 'own_mlp'
        if not own:
            if (edge_index is None or degree is None or degree.shape != (len(h),)
                    or edge_index.ndim != 2 or edge_index.shape[0] != 2):
                raise ValueError('The complete physical graph is required for the neighborhood head')
        if not enabled:
            raw = h.new_zeros(len(h), self.cfg.n_proteins)
        elif own:
            raw = self.readout(h)
        else:
            context = torch.cat([h, neighbor_mean(h, edge_index, degree), qc], 1)
            self_context = torch.cat([h, h, qc], 1)
            raw = self.readout(context) - self.readout(self_context)
        if enabled:
            raw = raw * torch.sigmoid(self.gate(torch.cat([h, qc], 1)))
        zero = torch.zeros_like(raw)
        state, local = (raw, zero) if own else (zero, center_local(raw, degree))
        return {'state_delta': state, 'local_delta': local, 'local_raw': raw,
                'correction': self.gamma_state * state + self.gamma_local * local}

    def specification(self):
        return {'config': asdict(self.cfg), 'deployed_parameters': sum(p.numel() for p in self.parameters()),
                'auxiliary_direct_head_deployed': False, 'prototype_or_sheaf_parameters': 0}

    @classmethod
    def from_research_checkpoint(cls, saved):
        original = saved['model_config']
        if (original.get('use_state') or original.get('use_niche') or original.get('local_operator') != 'none'
                or original.get('residual_head') not in {'own_mlp', 'neighbor_mlp'}):
            raise ValueError('A registered ordinary residual checkpoint is required')
        values = {key: original[key] for key in ResidualConfig.__dataclass_fields__ if key in original}
        policy = saved.get('execution_policy')
        if policy is None or 'weight_modulated_input' not in policy:
            raise ValueError('Checkpoint must declare its actual ensemble execution policy')
        values['weight_modulated_input'] = policy['weight_modulated_input']
        model = cls(ResidualConfig(**values))
        translated = {}
        discarded = []
        for key, value in saved['state_dict'].items():
            new = key
            for old, replacement in [('representations.rna.', 'encoder.'),
                                     ('residual_readout.', 'readout.'), ('residual_gate.', 'gate.')]:
                if key.startswith(old):
                    new = replacement + key[len(old):]
                    break
            if new in model.state_dict():
                translated[new] = value
            elif key.startswith('direct.') or key in {'anchor.weight', 'anchor.bias', 'spatial_alpha'}:
                discarded.append(key)
            else:
                raise ValueError('Unrecognized research tensor would be dropped: ' + key)
        model.load_state_dict(translated, strict=True)
        for name in ('gamma_state', 'gamma_local'):
            if not torch.all((getattr(model, name) == 0) | (getattr(model, name) == 1)):
                raise ValueError('Reference-selected gates must be binary')
        return model.eval(), {'discarded_auxiliary_or_external_base_tensors': discarded,
                              'retained_tensors': sorted(translated), **model.specification()}
