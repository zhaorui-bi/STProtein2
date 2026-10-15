"""Measured-protein scoring with explicit undefined-correlation coverage."""
import numpy as np
from scipy.stats import spearmanr


def evaluate(truth, prediction, proteins, edge_index=None):
    """Score aligned spot × protein matrices; complete-panel means never drop channels."""
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    proteins = tuple(proteins)
    if (y.shape != p.shape or y.ndim != 2 or not all(y.shape)
            or y.shape[1] != len(proteins) or len(set(proteins)) != len(proteins)):
        raise ValueError("Nonempty, uniquely named target/prediction shapes must match")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite values cannot be silently dropped")
    error = p - y
    rows = []
    for j, protein in enumerate(proteins):
        valid = np.ptp(y[:, j]) > 1e-12 and np.ptp(p[:, j]) > 1e-12
        rows.append({
            "protein": protein, "rmse": float(np.sqrt(np.mean(error[:, j] ** 2))),
            "mae": float(np.mean(np.abs(error[:, j]))),
            "pearson": float(np.corrcoef(y[:, j], p[:, j])[0, 1]) if valid else None,
            "spearman": float(spearmanr(y[:, j], p[:, j]).statistic) if valid else None,
            "mean_bias": float(error[:, j].mean()),
            "amplitude_ratio": float(p[:, j].std() / y[:, j].std()) if y[:, j].std() > 1e-12 else None,
            "correlation_defined": bool(valid),
        })
    complete = all(row['correlation_defined'] for row in rows)
    defined = [row['pearson'] for row in rows if row['correlation_defined']]
    yc, pc = y - y.mean(1, keepdims=True), p - p.mean(1, keepdims=True)
    valid_spots = (np.ptp(y, axis=1) > 1e-12) & (np.ptp(p, axis=1) > 1e-12)
    spot_mean = None
    if valid_spots.all():
        spot_mean = float(np.mean(np.clip(np.sum(yc * pc, axis=1) /
                                         np.sqrt(np.sum(yc ** 2, axis=1) * np.sum(pc ** 2, axis=1)), -1, 1)))
    summary = {
        "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.abs(error).mean()),
        "macro_protein_rmse": float(np.mean([row['rmse'] for row in rows])),
        "spots": len(y), "proteins": len(proteins),
        "undefined_correlations": len(rows) - len(defined),
        "mean_pearson": float(np.mean(defined)) if defined else None,
        "mean_pearson_complete_panel": float(np.mean([row['pearson'] for row in rows])) if complete else None,
        "mean_spearman_complete_panel": float(np.mean([row['spearman'] for row in rows])) if complete else None,
        "mean_spot_pearson": spot_mean,
        "undefined_spot_correlations": int((~valid_spots).sum()),
        "prediction_row_mean_abs": float(np.abs(p.mean(1)).mean()),
    }
    if edge_index is not None:
        edge = np.asarray(edge_index)
        if (edge.ndim != 2 or edge.shape[0] != 2 or not np.issubdtype(edge.dtype, np.integer)
                or (edge.size and (edge.min() < 0 or edge.max() >= len(y)))):
            raise ValueError("Integer graph edges must index prediction rows")
        if edge.shape[1]:
            i, j = edge
            summary['edge_difference_rmse'] = float(np.sqrt(np.mean(((p[i] - p[j]) - (y[i] - y[j])) ** 2)))
    return {"summary": summary, "per_protein": rows}
