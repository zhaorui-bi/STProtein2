"""RNA-only prediction with separately inspectable state and local corrections."""
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .prototypes import EnsembleLinear, Representations
from .sheaf import CooperativeSheaf


def center_local(raw, degree):
    eligible = (degree > 0).to(raw)
    mean = (raw * eligible[:, None]).sum(0) / eligible.sum().clamp_min(1)
    return eligible[:, None] * (raw - mean)


class GATExpert(nn.Module):
    """Ordinary single-head GAT control; receiver-normalized sparse attention."""
    def __init__(self, dim):
        super().__init__()
        self.value = nn.Linear(dim, dim, bias=False)
        self.receiver = nn.Linear(dim, 1, bias=False)
        self.sender = nn.Linear(dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        ids = torch.arange(len(h), device=h.device)
        row = torch.cat([edge_index[0], ids]); col = torch.cat([edge_index[1], ids])
        value = self.value(h)
        logits = F.leaky_relu(self.receiver(value)[row, 0] + self.sender(value)[col, 0], 0.2)
        maximum = logits.new_full((len(h),), -torch.inf).scatter_reduce(0, row, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - maximum[row])
        denominator = exp.new_zeros(len(h)).index_add(0, row, exp)
        alpha = exp / denominator[row]
        result = torch.zeros_like(value).index_add(0, row, alpha[:, None] * value[col])
        return self.norm(F.gelu(result)) - self.norm(F.gelu(value))


class STProtein2(nn.Module):
    def __init__(self, cfg=ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.representations = Representations(cfg)
        self.direct = EnsembleLinear(cfg.hidden, cfg.n_proteins, cfg.members)
        self.protein_embedding = nn.Parameter(torch.randn(cfg.n_proteins, cfg.protein_embedding) * 0.1)
        self.register_buffer("gamma_state", torch.ones(cfg.n_proteins) if cfg.use_state else torch.zeros(cfg.n_proteins))
        has_local = cfg.local_operator != "none" or cfg.use_niche
        self.register_buffer("gamma_local", torch.ones(cfg.n_proteins) if has_local else torch.zeros(cfg.n_proteins))
        dim = cfg.stalk * cfg.channels
        if cfg.use_state:
            self.state_readout = nn.Sequential(nn.Linear(cfg.hidden + cfg.prototype_dim, cfg.hidden),
                                               nn.GELU(), nn.Linear(cfg.hidden, cfg.n_proteins, bias=False))
            self.state_gate = nn.Linear(cfg.hidden + cfg.qc_dim, cfg.n_proteins)
            nn.init.zeros_(self.state_readout[-1].weight)
            nn.init.constant_(self.state_gate.bias, -2)
        else:
            self.state_readout = self.state_gate = None
        self.project = nn.Linear(cfg.hidden, dim) if cfg.local_operator != "none" else None
        if cfg.local_operator == "csnn":
            self.experts = nn.ModuleList([CooperativeSheaf(cfg.stalk, cfg.channels) for _ in range(cfg.experts)])
        elif cfg.local_operator == "gat":
            self.experts = nn.ModuleList([GATExpert(dim) for _ in range(cfg.experts)])
        else:
            self.experts = nn.ModuleList()
        self.expert_readout = nn.Linear(dim, cfg.n_proteins, bias=False) if self.experts else None
        if self.expert_readout is not None:
            nn.init.zeros_(self.expert_readout.weight)
        conditioning_dim = cfg.hidden + 2 * cfg.prototype_dim
        self.mixture = nn.Linear(conditioning_dim, cfg.experts * cfg.protein_embedding) if self.experts else None
        self.niche_readout = nn.Sequential(nn.Linear(cfg.hidden + cfg.prototype_dim, cfg.hidden),
                                          nn.GELU(), nn.Linear(cfg.hidden, cfg.n_proteins, bias=False)) if cfg.use_niche else None
        if self.niche_readout is not None:
            nn.init.zeros_(self.niche_readout[-1].weight)
        self.local_gate = nn.Linear(cfg.hidden + cfg.qc_dim, cfg.n_proteins) if has_local else None
        if self.local_gate is not None:
            nn.init.constant_(self.local_gate.bias, -2)

    @torch.no_grad()
    def set_gates(self, state, local):
        for name, value in (("gamma_state", state), ("gamma_local", local)):
            target = getattr(self, name)
            source = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if source.shape != target.shape or not torch.all((source == 0) | (source == 1)):
                raise ValueError("Gates must be binary with one entry per protein")
            target.copy_(source)

    def _graph_context(self, h, graph, partitions):
        if not self.experts:
            return None
        z = F.gelu(self.project(h))
        contexts = []
        for expert in self.experts:
            if self.cfg.local_operator == "gat":
                # Baseline uses full sparse graph; not advertised as a partitioned GAT kernel.
                contexts.append(expert(z, graph.edge_index))
                continue
            transformed, coeff, own = expert.prepare(z, graph.edge_index)
            if partitions is None:
                contexts.append(expert.correction(z, transformed, coeff, own, graph.edge_index))
                continue
            outputs, ownership = [], []
            for part in partitions:
                if isinstance(part[0], torch.Tensor):
                    core, nodes, local, edge = part
                else:
                    raise TypeError("Pass device-converted partition tuples")

                def run_block(a, b, *values, module=expert, nodes_=nodes, local_=local, edge_=edge):
                    return module.correction(a, b, values[:6], values[6:], edge_, nodes_, local_)

                inputs = (z, transformed, *coeff, *own)
                if self.training and self.cfg.checkpoint_blocks:
                    output = checkpoint(run_block, *inputs, use_reentrant=False)
                else:
                    output = run_block(*inputs)
                outputs.append(output); ownership.append(core)
            ids = torch.cat(ownership)
            if len(ids) != len(h) or not torch.equal(torch.sort(ids).values, torch.arange(len(h), device=h.device)):
                raise ValueError("Invalid unique Core output ownership")
            contexts.append(torch.cat(outputs)[torch.argsort(ids)])
        return torch.stack(contexts, dim=1)

    def forward(self, x, availability, qc, graph, partitions=None, enable_corrections=True):
        if x.ndim != 2 or x.shape[1] != self.cfg.n_genes or graph.n != len(x):
            raise ValueError("RNA dimensions do not match model and graph")
        if qc.shape != (len(x), self.cfg.qc_dim):
            raise ValueError("Invalid RNA QC shape")
        rep = self.representations(x, availability, qc, graph)
        h = rep["h"]
        base_members = self.direct(rep["members"])
        base = base_members.mean(1)
        zeros = h.new_zeros(len(h), self.cfg.prototype_dim)
        state_memory = rep["state"]["memory"] if rep["state"] is not None else zeros
        niche_memory = rep["niche"]["memory"] if rep["niche"] is not None else zeros
        state_delta = torch.zeros_like(base)
        local_raw = torch.zeros_like(base)
        if enable_corrections and self.state_readout is not None:
            contrast = self.state_readout(torch.cat([h, state_memory], 1)) - self.state_readout(torch.cat([h, zeros], 1))
            state_delta = torch.sigmoid(self.state_gate(torch.cat([h, qc], 1))) * contrast
        if enable_corrections and self.experts:
            context = self._graph_context(h, graph, partitions)
            prediction = self.expert_readout(context)  # N x experts x proteins
            node_queries = self.mixture(torch.cat([h, state_memory, niche_memory], 1)).reshape(len(h), self.cfg.experts, -1)
            if self.cfg.protein_conditioning:
                logits = torch.einsum("ned,pd->nep", node_queries, self.protein_embedding)
            else:
                logits = node_queries.mean(-1, keepdim=True).expand_as(prediction)
            local_raw = (torch.softmax(logits, dim=1) * prediction).sum(1)
        if enable_corrections and self.niche_readout is not None:
            own = rep["own_niche"]["memory"]
            local_raw = local_raw + self.niche_readout(torch.cat([h, niche_memory], 1)) - self.niche_readout(torch.cat([h, own], 1))
        if self.local_gate is not None:
            local_raw = local_raw * torch.sigmoid(self.local_gate(torch.cat([h, qc], 1)))
        local_delta = center_local(local_raw, graph.degree) if self.cfg.center_local else local_raw * (graph.degree > 0)[:, None]
        correction = self.gamma_state * state_delta + self.gamma_local * local_delta
        members = base_members + correction[:, None, :]
        return {"prediction": members.mean(1), "members": members, "base": base,
                "base_members": base_members, "state_delta": state_delta,
                "local_raw": local_raw, "local_delta": local_delta, "representations": rep}

    def architecture(self):
        return {"config": asdict(self.cfg), "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}

