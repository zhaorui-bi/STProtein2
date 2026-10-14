"""Measured-protein scoring is separate from RNA-only prediction."""
import numpy as np
from scipy.stats import spearmanr


def evaluate(truth, prediction, proteins, edge_index=None):
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(proteins):
        raise ValueError("Named target/prediction shapes must match")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite values cannot be silently dropped")
    error = p-y
    rows=[]
    for j, protein in enumerate(proteins):
        valid = np.ptp(y[:,j]) > 1e-12 and np.ptp(p[:,j]) > 1e-12
        rows.append({"protein":protein,"rmse":float(np.sqrt(np.mean(error[:,j]**2))),
                     "mae":float(np.mean(np.abs(error[:,j]))),
                     "pearson":float(np.corrcoef(y[:,j],p[:,j])[0,1]) if valid else None,
                     "spearman":float(spearmanr(y[:,j],p[:,j]).statistic) if valid else None,
                     "mean_bias":float(error[:,j].mean()),
                     "amplitude_ratio":float(p[:,j].std()/y[:,j].std()) if y[:,j].std()>1e-12 else None,
                     "correlation_defined":bool(valid)})
    summary={"rmse":float(np.sqrt(np.mean(error**2))),"mae":float(np.abs(error).mean()),
             "spots":len(y),"proteins":len(proteins),"undefined_correlations":sum(not r['correlation_defined'] for r in rows),
             "mean_pearson":float(np.mean([r['pearson'] for r in rows if r['pearson'] is not None])) if any(r['correlation_defined'] for r in rows) else None,
             "prediction_row_mean_abs":float(np.abs(p.mean(1)).mean())}
    if edge_index is not None and edge_index.shape[1]:
        i,j=edge_index
        summary['edge_difference_rmse']=float(np.sqrt(np.mean(((p[i]-p[j])-(y[i]-y[j]))**2)))
    return {"summary":summary,"per_protein":rows}

