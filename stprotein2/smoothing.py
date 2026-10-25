"""Mean-preserving spatial basis used by model training and inference."""
import numpy as np


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
