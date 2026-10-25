"""Canonical physical graph; no routing or partition execution."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

def digest_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

@dataclass
class SpatialGraph:
    n: int
    edge_index: np.ndarray
    coordinates: np.ndarray
    distance_scale: float

    def __post_init__(self):
        e = np.asarray(self.edge_index, dtype=np.int64)
        if e.ndim != 2 or e.shape[0] != 2 or (e.size and (e.min() < 0 or e.max() >= self.n)):
            raise ValueError('Invalid edge IDs')
        if e.size and np.any(e[0] == e[1]):
            raise ValueError('Physical graph excludes self edges')
        if e.shape[1] != len(set(map(tuple, e.T))):
            raise ValueError('Duplicate physical edges')
        self.edge_index = e
        self.adjacency = sparse.csr_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(self.n, self.n))
        self.degree = np.asarray(self.adjacency.sum(1)).ravel()
        self.fingerprint = digest_array(e)

def physical_graph(coordinates, k=6, radius=2.5):
    c = np.asarray(coordinates, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 1 or (not np.isfinite(c).all()):
        raise ValueError('Require nonempty finite N x 2 coordinates')
    if k < 1 or radius <= 0:
        raise ValueError('Invalid graph settings')
    n = len(c)
    if n == 1:
        return SpatialGraph(1, np.empty((2, 0), dtype=np.int64), c, 1.0)
    (distances, neighbors) = cKDTree(c).query(c, k=min(k + 1, n))
    positive = np.where(distances > 0, distances, np.inf).min(1)
    finite = positive[np.isfinite(positive)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    pairs = set()
    for i in range(n):
        choices = sorted(((float(d), int(j)) for (d, j) in zip(distances[i], neighbors[i]) if j != i))
        for (distance, j) in choices[:k]:
            if distance <= radius * scale:
                pairs.add((i, j))
                pairs.add((j, i))
    edge = np.asarray(sorted(pairs), dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    return SpatialGraph(n, edge, c, scale)
