"""Shared-member RNA representation and matching student/EMA prototype slots."""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans


class EnsembleLinear(nn.Module):
    """BatchEnsemble-style layer, independently supervised per member."""
    def __init__(self, n_in, n_out, members):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.r = nn.Parameter(torch.empty(members, n_in))
        self.s = nn.Parameter(torch.empty(members, n_out))
        self.bias = nn.Parameter(torch.zeros(members, n_out))
        nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
        with torch.no_grad():
            self.r.bernoulli_(0.5).mul_(2).sub_(1)
            self.s.fill_(1)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        return ((x * self.r) @ self.weight) * self.s + self.bias


class RNAEnsemble(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EnsembleLinear(cfg.n_genes, cfg.width, cfg.members),
                                     EnsembleLinear(cfg.width, cfg.hidden, cfg.members)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.hidden)])
        self.mask = nn.Sequential(nn.Linear(cfg.n_genes, 32), nn.GELU(), nn.Linear(32, cfg.hidden))
        self.qc = nn.Linear(cfg.qc_dim, cfg.hidden)
        self.output_norm = nn.LayerNorm(cfg.hidden)
        self.dropout = cfg.dropout
        self.use_encoder_qc = cfg.use_encoder_qc

    def forward(self, x, availability, qc):
        if availability.ndim == 1:
            availability = availability[None, :].expand_as(x)
        if availability.shape != x.shape:
            raise ValueError("Availability must match input genes")
        # Unavailable values cannot influence the encoder, even if callers retain
        # a nonzero placeholder. Observed values must be finite at the data boundary.
        h = torch.where(availability > 0, x, torch.zeros_like(x))
        for linear, norm in zip(self.layers, self.norms):
            h = F.dropout(F.gelu(norm(linear(h))), p=self.dropout, training=self.training)
        mask_h = self.mask(1 - availability)
        encoder_qc = qc if self.use_encoder_qc else torch.zeros_like(qc)
        h = self.output_norm(h + mask_h[:, None, :] + self.qc(encoder_qc)[:, None, :])
        return h


class PrototypeBank(nn.Module):
    def __init__(self, n_in, n_codes, dimension, temperature):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(n_in, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.codes = nn.Parameter(F.normalize(torch.randn(n_codes, dimension), dim=1))
        self.temperature = temperature

    def forward(self, h):
        z = F.normalize(self.project(h), dim=1)
        codes = F.normalize(self.codes, dim=1)
        logits = z @ codes.T
        q = torch.softmax(logits / self.temperature, dim=1)
        return {"z": z, "logits": logits, "q": q, "memory": q @ codes}

    @torch.no_grad()
    def initialize(self, z, seed=2025):
        if len(z) < len(self.codes):
            raise ValueError("Too few reference points to initialize prototypes")
        a = z.detach().cpu().numpy()
        centers = KMeans(len(self.codes), n_init=5, random_state=seed).fit(a).cluster_centers_
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        self.codes.copy_(torch.as_tensor(centers, device=self.codes.device, dtype=self.codes.dtype))


class Representations(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rna = RNAEnsemble(cfg)
        self.state = PrototypeBank(cfg.hidden, cfg.state_prototypes, cfg.prototype_dim,
                                   cfg.prototype_temperature) if cfg.use_state else None
        niche_in = 2 * cfg.hidden + cfg.state_prototypes + cfg.qc_dim
        self.niche = PrototypeBank(niche_in, cfg.niche_prototypes, cfg.prototype_dim,
                                   cfg.prototype_temperature) if cfg.use_niche else None

    def forward(self, x, availability, qc, graph):
        members = self.rna(x, availability, qc)
        h = members.mean(1)
        state = self.state(h) if self.state is not None else None
        q = state["q"] if state is not None else h.new_zeros(len(h), self.cfg.state_prototypes)
        niche = own_niche = None
        if self.niche is not None:
            niche = self.niche(torch.cat([h, graph.mean(h), graph.mean(q), qc], dim=1))
            own_niche = self.niche(torch.cat([h, h, q, qc], dim=1))
        return {"members": members, "h": h, "state": state, "niche": niche, "own_niche": own_niche}


class EMATeacher(nn.Module):
    """Only representation modules; no sheaf or protein decoder computation."""
    def __init__(self, student_representations, center_momentum=0.9):
        super().__init__()
        self.representations = copy.deepcopy(student_representations).requires_grad_(False)
        cfg = student_representations.cfg
        self.register_buffer("state_center", torch.zeros(1, cfg.state_prototypes))
        self.register_buffer("niche_center", torch.zeros(1, cfg.niche_prototypes))
        self.center_momentum = center_momentum
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, availability, qc, graph):
        return self.representations(x, availability, qc, graph)

    @torch.no_grad()
    def update(self, student_representations, momentum):
        source = dict(student_representations.named_parameters())
        for name, parameter in self.representations.named_parameters():
            parameter.lerp_(source[name].detach(), 1 - momentum)
        source_buffers = dict(student_representations.named_buffers())
        for name, value in self.representations.named_buffers():
            value.copy_(source_buffers[name])
        self.update_count.add_(1)

    def loss(self, student, teacher, temperature):
        losses = {}
        for name in ("state", "niche"):
            s, t = student[name], teacher[name]
            if s is None:
                losses[name] = student["h"].new_zeros(())
                continue
            center = getattr(self, name + "_center")
            target = torch.softmax((t["logits"] - center) / temperature, dim=1).detach()
            logits = s["logits"] / self.representations.cfg.prototype_temperature
            losses[name] = -(target * F.log_softmax(logits, dim=1)).sum(1).mean() / math.log(logits.shape[1])
        return losses

    @torch.no_grad()
    def update_centers(self, teacher):
        # Caller supplies exactly one row per whole-slice node, not halo copies.
        for name in ("state", "niche"):
            if teacher[name] is not None:
                center = getattr(self, name + "_center")
                center.lerp_(teacher[name]["logits"].mean(0, keepdim=True), 1 - self.center_momentum)


@torch.no_grad()
def prototype_diagnostics(bank_output, bank):
    if bank_output is None:
        return {}
    q = bank_output["q"].clamp_min(1e-12)
    mean = q.mean(0)
    codes = F.normalize(bank.codes, dim=1)
    cosine = codes @ codes.T
    off_diagonal = ~torch.eye(len(codes), device=q.device, dtype=torch.bool)
    return {"effective_codes": float(torch.exp(-(mean * mean.log()).sum())),
            "mean_entropy": float(-(q * q.log()).sum(1).mean()),
            "maximum_offdiagonal_cosine": float(cosine[off_diagonal].max()) if len(codes) > 1 else 1.0,
            "occupancy": mean.cpu().tolist()}

