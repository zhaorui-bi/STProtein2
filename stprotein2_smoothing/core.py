"""Label-free graph residual and one shared, reference-selected coefficient."""
import hashlib

import numpy as np

from stprotein2.graph import SpatialGraph, digest_array


ALPHAS = (0., .25, .5, .75, 1.)
MODES = ('physical', 'permuted')


def permute_graph(graph, barcodes, sample, seed=2025):
    """Relabel a fixed graph; retain its topology and degree multiset.

    Individual spot degrees need not be preserved. Canonical barcode order
    makes the same logical permutation independent of input row numbering.
    """
    barcodes = np.asarray(barcodes, dtype=str)
    if barcodes.shape != (graph.n,) or len(np.unique(barcodes)) != graph.n:
        raise ValueError('Unique barcodes required for graph relabeling')
    salt = int.from_bytes(hashlib.sha256(str(sample).encode()).digest()[:4], 'little')
    order = np.argsort(barcodes, kind='stable')
    rng = np.random.default_rng([seed, salt])
    permutation = np.empty(graph.n, dtype=np.int64)
    permutation[order] = order[rng.permutation(graph.n)]
    edge = permutation[graph.edge_index]
    edge = edge[:, np.lexsort((edge[1], edge[0]))]
    return SpatialGraph(graph.n, edge, graph.coordinates.copy(), graph.distance_scale), permutation


def spatial_residual(base, graph):
    """Neighbor mean minus self, centered over nonisolated spots per protein."""
    base = np.asarray(base)
    if base.ndim != 2 or len(base) != graph.n or not np.issubdtype(base.dtype, np.floating) or not np.isfinite(base).all():
        raise ValueError('Finite floating-point node-by-protein predictions required')
    value = base.astype(np.float64)
    valid = graph.degree > 0
    result = np.zeros_like(value)
    if valid.any():
        result[valid] = (graph.adjacency @ value)[valid] / graph.degree[valid, None] - value[valid]
        result[valid] -= result[valid].mean(axis=0, keepdims=True)
    return result


def apply(base, residual, alpha):
    base, residual = np.asarray(base), np.asarray(residual)
    if (base.ndim != 2 or not np.issubdtype(base.dtype, np.floating) or not np.isfinite(base).all()
            or base.shape != residual.shape or not np.isfinite(residual).all()
            or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('Aligned finite residual and coefficient in [0,1] required')
    if alpha == 0:
        return base.copy()
    return (base.astype(np.float64) + float(alpha) * residual).astype(base.dtype)


def select_alpha(records, alphas=ALPHAS):
    """One coefficient across all proteins; equal source / equal slice MSE.

    Records must be genuinely held-source predictions. Provenance isolation is
    enforced by the experiment runner. This is baseline hyperparameter CV,
    separate from the neural model's predeclared per-protein gate rule.
    """
    if not records or len({r['name'] for r in records}) != len(records):
        raise ValueError('Distinct held-source slices required')
    grid = tuple(float(a) for a in alphas)
    if not grid or not np.isfinite(grid).all() or grid != tuple(sorted(set(grid))) or grid[0] != 0 or grid[-1] > 1:
        raise ValueError('Sorted unique nonnegative grid including zero required')
    by_slice = {}
    for row in records:
        if row['base'].shape != row['target'].shape or row['target'].ndim != 2 or not np.isfinite(row['target']).all():
            raise ValueError('Finite aligned target required for reference selection')
        by_slice[row['name']] = [float(np.mean((apply(row['base'], row['residual'], a).astype(np.float64) - row['target'])**2))
                                 for a in grid]
    groups = sorted({r['group'] for r in records})
    by_group = {g: np.mean([by_slice[r['name']] for r in records if r['group'] == g], axis=0).tolist() for g in groups}
    scores = np.mean(list(by_group.values()), axis=0)
    # Numerical ties prefer less correction, including exact anchor fallback.
    selected = int(np.flatnonzero(scores <= scores.min() + 1e-12)[0])
    return {'alpha': grid[selected], 'grid': list(grid), 'mean_group_mse': scores.tolist(),
            'group_mse': by_group, 'slice_mse': by_slice, 'selection_groups': groups,
            'rule': 'equal-group/equal-slice MSE; ties within1e-12 choose smaller alpha',
            'records': [{'name': r['name'], 'group': r['group'],
                         'base_sha256': digest_array(r['base']), 'residual_sha256': digest_array(r['residual']),
                         'target_sha256': digest_array(r['target'])} for r in records]}
