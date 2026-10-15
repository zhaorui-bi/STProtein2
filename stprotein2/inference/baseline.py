"""RNA-only inference for a fitted spatial Ridge baseline."""
import numpy as np

from stprotein2.data import PANEL
from stprotein2.graph import physical_graph
from stprotein2.training.residual_targets import predict_anchor
from stprotein2.models.smoothing import spatial_residual, apply


def predict(bundle, view, alpha, mode='physical', seed=2025):
    if mode != 'physical':
        raise ValueError('Deployment requires a physical graph')
    base = predict_anchor(bundle, view)
    graph = physical_graph(view.coordinates, k=6, radius=2.5)
    residual = spatial_residual(base, graph)
    return {'prediction': apply(base, residual, alpha), 'base': base, 'local_delta': residual,
            'barcodes': view.barcodes, 'proteins': np.asarray(PANEL),
            'graph_sha256': graph.fingerprint,
            'permutation_sha256': None,
            'isolated_nodes': int(np.sum(graph.degree == 0))}
