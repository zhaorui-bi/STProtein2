"""Supplementary-paper metrics, explicit diagnostics, and strict paired coverage."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score


PAPER_DIRECTIONS = {
    'rmse': 'lower', 'mae': 'lower', 'macro_protein_rmse': 'lower',
    'mean_pearson_complete_panel': 'higher', 'mean_spot_pearson': 'higher',
    'mean_spearman_complete_panel': 'higher',
    'moran_alignment_MAE': 'lower', 'moran_alignment_RMSE': 'lower', 'moran_alignment_corr': 'higher',
    'geary_alignment_MAE': 'lower', 'geary_alignment_RMSE': 'lower', 'geary_alignment_corr': 'higher',
    'residual_moran_mean': 'diagnostic', 'residual_moran_median': 'diagnostic',
    'ari': 'higher', 'nmi': 'higher', 'domain_homogeneity': 'higher', 'domain_completeness': 'higher',
}


def spatial_statistics(values, graph):
    """Binary directed-edge Moran I and Geary C, excluding constant channels."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != graph.n or not np.isfinite(x).all():
        raise ValueError('Finite N x P spatial values required')
    row, col = graph.edge_index
    z = x - x.mean(0)
    denominator = np.sum(z * z, axis=0)
    valid = denominator > 1e-24
    if graph.n < 2 or not len(row):
        return {'moran': [None] * x.shape[1], 'geary': [None] * x.shape[1]}
    numer_m = np.sum(z[row] * z[col], axis=0)
    numer_g = np.sum((x[row] - x[col]) ** 2, axis=0)
    m = np.divide(graph.n * numer_m, len(row) * denominator, out=np.zeros_like(denominator), where=valid)
    g = np.divide((graph.n - 1) * numer_g, 2 * len(row) * denominator, out=np.zeros_like(denominator), where=valid)
    return {'moran': [float(v) if ok else None for v, ok in zip(m, valid)],
            'geary': [float(v) if ok else None for v, ok in zip(g, valid)]}


def agreement(truth, prediction):
    if any(v is None for v in truth + prediction):
        return {'MAE': None, 'RMSE': None, 'corr': None}
    a, b = np.asarray(truth), np.asarray(prediction)
    correlation = None if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12 else float(pearsonr(a, b).statistic)
    return {'MAE': float(np.mean(np.abs(a-b))), 'RMSE': float(np.sqrt(np.mean((a-b)**2))), 'corr': correlation}


def extended_metrics(truth, prediction, graph, cluster_counts=(4, 6, 8)):
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Matching finite protein matrices required')
    measured, predicted, residual = [spatial_statistics(v, graph) for v in (y, p, p-y)]
    result = {}
    for statistic in ('moran', 'geary'):
        for key, value in agreement(measured[statistic], predicted[statistic]).items():
            result[statistic + '_alignment_' + key] = value
    valid = [v for v in residual['moran'] if v is not None]
    complete = len(valid) == y.shape[1]
    result.update(residual_moran_mean=float(np.mean(valid)) if complete else None,
                  residual_moran_median=float(np.median(valid)) if complete else None,
                  residual_moran_abs_expected_deviation=float(np.mean(np.abs(np.asarray(valid) + 1/(graph.n-1))))
                      if complete and graph.n > 1 else None,
                  undefined_spatial_channels=sum(v is None for v in measured['moran']) + sum(v is None for v in predicted['moran']),
                  undefined_residual_spatial_channels=y.shape[1]-len(valid))
    for k in cluster_counts:
        if not 2 <= k < len(y):
            raise ValueError('Invalid prespecified cluster count')
        a = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(y)
        b = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(p)
        scores = {'ari': float(adjusted_rand_score(a, b)), 'nmi': float(normalized_mutual_info_score(a, b)),
                  'domain_homogeneity': float(homogeneity_score(a, b)),
                  'domain_completeness': float(completeness_score(a, b))}
        result.update({f'cluster_k{k}_{key}': value for key, value in scores.items()})
        if k == 6:
            result.update(scores)
    return {'summary': result, 'per_protein_spatial': {'truth': measured, 'prediction': predicted, 'residual': residual}}


def compare(reference, candidate, expected_queries, expected_seeds, keys=PAPER_DIRECTIONS, tolerance=1e-10):
    """Never pool partial panels or choose favorable seeds; preserve every failed cell."""
    expected = {(q, s) for q in expected_queries for s in expected_seeds}
    def indexed(rows):
        result = {(r['query'], r['seed']): r for r in rows}
        if len(result) != len(rows) or set(result) != expected:
            raise ValueError('Missing, duplicate or unexpected paired query/seed coverage')
        return result
    a, b = indexed(reference), indexed(candidate)
    cells = []
    for query in expected_queries:
        for metric, direction in keys.items():
            av = [a[query, s][metric] for s in expected_seeds]
            bv = [b[query, s][metric] for s in expected_seeds]
            status = 'undefined' if any(v is None or not np.isfinite(v) for v in av+bv) else 'diagnostic'
            am, bm = (None, None) if status == 'undefined' else (float(np.mean(av)), float(np.mean(bv)))
            if status != 'undefined' and direction != 'diagnostic':
                gain = (bm-am) * (1 if direction == 'higher' else -1)
                status = 'win' if gain > tolerance else ('loss' if gain < -tolerance else 'tie')
            cells.append({'query': query, 'metric': metric, 'direction': direction, 'status': status,
                          'dpas_mean': am, 'candidate_mean': bm,
                          'dpas_seed_sd': None if am is None else float(np.std(av, ddof=1)) if len(av)>1 else 0.,
                          'candidate_seed_sd': None if bm is None else float(np.std(bv, ddof=1)) if len(bv)>1 else 0.,
                          'seed_differences': [None if x is None or z is None else z-x for x,z in zip(av,bv)]})
    quality = [r for r in cells if r['direction'] != 'diagnostic']
    return {'cells': cells, 'counts': {s: sum(r['status']==s for r in quality) for s in ('win','loss','tie','undefined')},
            'all_quality_cells_strictly_won': bool(quality) and all(r['status']=='win' for r in quality)}
