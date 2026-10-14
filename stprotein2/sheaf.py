"""Cooperative sheaf propagation with complete-graph normalization.

Independent PyTorch implementation of the two normalized propagation operators
in CSNN (ICLR 2026). The audited reference is recorded in UPSTREAM.json. We use
node-wise Cayley maps, an option in the reference, without PyG/extension kernels.
This module contains no copied upstream source and claims no inherited theorem.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SheafCoefficients:
    t_out: torch.Tensor
    s_out: torch.Tensor
    t_in: torch.Tensor
    s_in: torch.Tensor
    diag_out: torch.Tensor
    diag_in: torch.Tensor

    def tensors(self):
        return (self.t_out, self.s_out, self.t_in, self.s_in, self.diag_out, self.diag_in)


def normalized_coefficients(t, s, ct, cs, edge_index, n):
    """All coefficients use the COMPLETE view graph, before any partition."""
    row, col = edge_index
    outgoing = torch.bincount(row, minlength=n).to(ct)
    incoming = torch.bincount(col, minlength=n).to(cs)
    d_out = outgoing * ct.square()
    d_in = incoming * cs.square()
    out_inv = torch.rsqrt(1 + d_out)
    in_inv = torch.rsqrt(1 + d_in)
    t_scaled = t * ct[:, None, None]
    s_scaled = s * cs[:, None, None]
    return SheafCoefficients(
        t_scaled * out_inv[:, None, None], s_scaled * out_inv[:, None, None],
        t_scaled * in_inv[:, None, None], s_scaled * in_inv[:, None, None],
        outgoing * ct.square() / (1 + d_out),
        outgoing * cs.square() / (1 + d_in),
    )


def propagate(x, edge_index, receiver_maps, sender_maps, diagonal):
    row, col = edge_index
    sent = sender_maps @ x
    aggregated = torch.zeros_like(x).index_add(0, row, sent[col])
    return diagonal[:, None, None] * x - receiver_maps.transpose(-1, -2) @ aggregated


def cooperative_operator(x, edge_index, coefficients):
    t_out, s_out, t_in, s_in, diag_out, diag_in = coefficients
    first = propagate(x, edge_index, t_out, s_out, diag_out)
    return propagate(first, edge_index, t_in, s_in, diag_in)


class NodeConformalMaps(nn.Module):
    def __init__(self, features, stalk):
        super().__init__()
        self.stalk = stalk
        self.network = nn.Sequential(nn.Linear(features, features), nn.GELU(),
                                     nn.Linear(features, stalk * (stalk - 1) // 2 + 1))
        indices = torch.tril_indices(stalk, stalk, -1)
        self.register_buffer("lower_rows", indices[0])
        self.register_buffer("lower_cols", indices[1])

    def forward(self, x):
        values = self.network(x)
        lower = x.new_zeros(len(x), self.stalk, self.stalk)
        lower[:, self.lower_rows, self.lower_cols] = torch.tanh(values[:, :-1])
        skew = lower - lower.transpose(-1, -2)
        eye = torch.eye(self.stalk, device=x.device, dtype=x.dtype)
        q = torch.linalg.solve(eye - 0.5 * skew, eye + 0.5 * skew)
        # Upper cap follows reference; lower cap is an explicit numerical adaptation.
        scale = torch.exp(values[:, -1].clamp(-6.0, math.log(10.0)))
        return q, scale


class CooperativeSheaf(nn.Module):
    def __init__(self, stalk=8, channels=8):
        super().__init__()
        self.stalk, self.channels = stalk, channels
        dim = stalk * channels
        self.target_map = NodeConformalMaps(dim, stalk)
        self.source_map = NodeConformalMaps(dim, stalk)
        self.left = nn.Parameter(torch.eye(stalk))
        self.right = nn.Parameter(torch.empty(channels, channels))
        nn.init.orthogonal_(self.right)
        self.epsilon = nn.Parameter(torch.zeros(stalk, 1))
        self.norm = nn.LayerNorm(dim)

    def prepare(self, h, edge_index):
        t, ct = self.target_map(h)
        s, cs = self.source_map(h)
        x = h.reshape(len(h), self.stalk, self.channels)
        transformed = self.left @ x @ self.right
        full = normalized_coefficients(t, s, ct, cs, edge_index, len(h))
        identity_ids = torch.arange(len(h), device=h.device)
        self_edges = torch.stack([identity_ids, identity_ids])
        own = normalized_coefficients(t, s, ct, cs, self_edges, len(h))
        return transformed, full.tensors(), own.tensors()

    def _finish(self, h, propagated):
        original = h.reshape(len(h), self.stalk, self.channels)
        value = (1 + torch.tanh(self.epsilon)) * original - F.gelu(propagated)
        return self.norm(value.flatten(1))

    def correction(self, h, transformed, coefficients, self_coefficients, edge_index, nodes=None, core_local=None):
        if nodes is not None:
            h, transformed = h[nodes], transformed[nodes]
            coefficients = tuple(c[nodes] for c in coefficients)
            self_coefficients = tuple(c[nodes] for c in self_coefficients)
        output = cooperative_operator(transformed, edge_index, coefficients)
        ids = torch.arange(len(h), device=h.device)
        own = cooperative_operator(transformed, torch.stack([ids, ids]), self_coefficients)
        result = self._finish(h, output) - self._finish(h, own)
        return result if core_local is None else result[core_local]

