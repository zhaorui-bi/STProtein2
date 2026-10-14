"""Sparse physical graphs and unique Core/complete Halo ownership.

edge_index[0] is RECEIVER, edge_index[1] is SENDER throughout this package.
No label is accepted by any graph or partition function.
"""
from dataclasses import dataclass
import hashlib
import heapq
from collections import deque

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch


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
            raise ValueError("Invalid edge IDs")
        if e.size and np.any(e[0] == e[1]):
            raise ValueError("Physical graph excludes self edges")
        if e.shape[1] != len(set(map(tuple, e.T))):
            raise ValueError("Duplicate physical edges")
        self.edge_index = e
        self.adjacency = sparse.csr_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(self.n, self.n))
        self.degree = np.asarray(self.adjacency.sum(1)).ravel()
        self.fingerprint = digest_array(e)

    def torch(self, device="cpu", dtype=torch.float32):
        return TorchGraph(torch.as_tensor(self.edge_index, device=device, dtype=torch.long),
                          torch.as_tensor(self.degree, device=device, dtype=dtype), self.n)


@dataclass
class TorchGraph:
    edge_index: torch.Tensor
    degree: torch.Tensor
    n: int

    def mean(self, x):
        if len(x) != self.n:
            raise ValueError("Node count mismatch")
        row, col = self.edge_index
        result = torch.zeros_like(x).index_add(0, row, x[col])
        shape = (self.n,) + (1,) * (x.ndim - 1)
        result = result / self.degree.clamp_min(1).reshape(shape)
        return torch.where((self.degree > 0).reshape(shape), result, x)


def physical_graph(coordinates, k=6, radius=2.5):
    c = np.asarray(coordinates, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 1 or not np.isfinite(c).all():
        raise ValueError("Require nonempty finite N x 2 coordinates")
    if k < 1 or radius <= 0:
        raise ValueError("Invalid graph settings")
    n = len(c)
    if n == 1:
        return SpatialGraph(1, np.empty((2, 0), dtype=np.int64), c, 1.0)
    distances, neighbors = cKDTree(c).query(c, k=min(k + 1, n))
    positive = np.where(distances > 0, distances, np.inf).min(1)
    finite = positive[np.isfinite(positive)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    # Explicitly remove self IDs (duplicate coordinates need not place self first).
    pairs = set()
    for i in range(n):
        choices = sorted((float(d), int(j)) for d, j in zip(distances[i], neighbors[i]) if j != i)
        for distance, j in choices[:k]:
            if distance <= radius * scale:
                pairs.add((i, j)); pairs.add((j, i))
    edge = np.asarray(sorted(pairs), dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
    return SpatialGraph(n, edge, c, scale)


@dataclass
class CoreHalo:
    core: np.ndarray
    nodes: np.ndarray
    core_local: np.ndarray
    edge_index: np.ndarray
    hops: int

    def to(self, device):
        return tuple(torch.as_tensor(x, device=device, dtype=torch.long)
                     for x in (self.core, self.nodes, self.core_local, self.edge_index))


def complete_halo(graph, core, hops=2):
    core = np.asarray(core, dtype=np.int64)
    if len(core) == 0 or len(np.unique(core)) != len(core) or core.min() < 0 or core.max() >= graph.n:
        raise ValueError("Invalid core ownership")
    if hops < 0:
        raise ValueError("Negative halo depth")
    included = np.zeros(graph.n, dtype=bool); included[core] = True
    frontier = core
    for _ in range(hops):
        neighbors = graph.adjacency[frontier].indices
        new = np.unique(neighbors[~included[neighbors]])
        included[new] = True
        frontier = new
    nodes = np.flatnonzero(included)
    mapping = np.full(graph.n, -1, dtype=np.int64); mapping[nodes] = np.arange(len(nodes))
    row, col = graph.edge_index
    take = included[row] & included[col]
    edge = np.stack([mapping[row[take]], mapping[col[take]]])
    return CoreHalo(core, nodes, mapping[core], edge, hops)


def validate_partition(graph, partitions, required_hops=2):
    ids = np.concatenate([p.core for p in partitions]) if partitions else np.empty(0, dtype=np.int64)
    if not np.array_equal(np.sort(ids), np.arange(graph.n)):
        raise ValueError("Every node must have exactly one core owner")
    for p in partitions:
        required = complete_halo(graph, p.core, required_hops)
        if not np.isin(required.nodes, p.nodes).all():
            raise ValueError("Truncated mandatory halo")
        expected = complete_halo(graph, p.core, p.hops)
        if not (np.array_equal(p.nodes, expected.nodes) and np.array_equal(p.edge_index, expected.edge_index)
                and np.array_equal(p.core_local, expected.core_local)):
            raise ValueError("Halo IDs or internal edges do not match the physical graph")
    return True


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class FrozenRouter:
    """Reference-fitted PCA and spherical centers, distinct from model codebooks."""
    def __init__(self, components=50, clusters=32, seed=2025, per_group=4000):
        self.components, self.clusters, self.seed, self.per_group = components, clusters, seed, per_group

    def fit(self, arrays, groups):
        if len(arrays) != len(groups) or not arrays:
            raise ValueError("Reference arrays and group IDs required")
        rng = np.random.default_rng(self.seed)
        samples = []
        pooled = {group: np.concatenate([x for x, g in zip(arrays, groups) if g == group])
                  for group in sorted(set(groups))}
        count = min(self.per_group, min(len(x) for x in pooled.values()))
        for combined in pooled.values():
            samples.append(combined[rng.choice(len(combined), count, replace=False)])
        data = np.concatenate(samples)
        d = min(self.components, data.shape[1], len(data) - 1)
        if d < 1:
            raise ValueError("Insufficient reference RNA for router")
        self.pca_ = PCA(n_components=d, svd_solver="randomized", random_state=self.seed).fit(data)
        z = _normalize(self.pca_.transform(data))
        k = min(self.clusters, len(z))
        centers = _normalize(KMeans(k, n_init=5, random_state=self.seed).fit(z).cluster_centers_)
        for _ in range(10):
            labels = (z @ centers.T).argmax(1)
            centers = _normalize(np.stack([z[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)]))
        self.centers_ = centers.astype(np.float32)
        self.fitted_groups_ = sorted(set(groups))
        self.samples_per_group_ = count
        return self

    def transform(self, x):
        z = _normalize(self.pca_.transform(x)).astype(np.float32)
        logits = z @ self.centers_.T / 0.2
        q = np.exp(logits - logits.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        return z, q


def boundary_score(graph, assignment):
    q = np.asarray(assignment, dtype=np.float64)
    if len(q) != graph.n or np.any(q < 0) or not np.allclose(q.sum(1), 1):
        raise ValueError("Router assignments must be normalized probabilities")
    if q.shape[1] <= 1:
        return np.zeros(graph.n)
    q = np.maximum(q, 1e-12)
    entropy = -(q * np.log(q)).sum(1) / np.log(q.shape[1])
    row, col = graph.edge_index
    mid = (q[row] + q[col]) / 2
    js = 0.5 * ((q[row] * np.log(q[row] / mid)).sum(1) + (q[col] * np.log(q[col] / mid)).sum(1)) / np.log(2)
    local = np.bincount(row, weights=js, minlength=graph.n) / np.maximum(graph.degree, 1)
    return np.clip(0.5 * (entropy + local), 0, 1)


def partition_graph(graph, core_size=256, score=None, embedding=None, min_core=128, max_core=512, hops=2,
                    node_keys=None, merge_small=True):
    """Deterministic connected region growth; never truncates required halos.

    Seeds maximize distance to existing seeds in graph hops, with stable barcode
    tie breaking. Small remainders merge only with physically adjacent cores
    under max_core; disconnected or unmergeable remainders may remain small.
    The node budget is a target, not permission to remove dependency nodes.
    """
    if not 1 <= min_core <= max_core or core_size < 1:
        raise ValueError("Invalid core budget")
    if score is not None and (np.shape(score) != (graph.n,) or not np.isfinite(score).all()):
        raise ValueError("Invalid boundary score")
    keys = np.asarray(node_keys if node_keys is not None else np.arange(graph.n))
    if keys.shape != (graph.n,) or len(np.unique(keys)) != graph.n:
        raise ValueError("Unique node keys required for partition tie breaking")
    order = np.argsort(keys, kind="stable")
    rank = np.empty(graph.n, dtype=np.int64); rank[order] = np.arange(graph.n)
    seed_distance = np.full(graph.n, np.inf)
    assigned = np.zeros(graph.n, dtype=bool)
    cores = []
    while not assigned.all():
        candidates = order[~assigned[order]]
        seed = int(candidates[np.argmax(seed_distance[candidates])])
        frontier = deque([seed]); seed_distance[seed] = 0
        while frontier:
            i = frontier.popleft()
            distance = seed_distance[i] + 1
            for j in graph.adjacency[i].indices:
                if distance < seed_distance[j]:
                    seed_distance[j] = distance; frontier.append(int(j))
        core, queue, queued = [], [(0.0, int(rank[seed]), seed)], {seed}
        target = core_size
        while queue and len(core) < target:
            _, _, i = heapq.heappop(queue)
            if assigned[i]:
                continue
            assigned[i] = True; core.append(i)
            if score is not None:
                target = int(np.clip(32 * round((512 / (1 + 3 * float(np.mean(score[core])))) / 32), min_core, max_core))
            for j in graph.adjacency[i].indices:
                j = int(j)
                if assigned[j] or j in queued:
                    continue
                d = np.linalg.norm(graph.coordinates[i] - graph.coordinates[j]) / graph.distance_scale
                expression = float(np.sum((embedding[i] - embedding[j]) ** 2)) if embedding is not None else 0.0
                affinity = np.exp(-d * d - expression)
                heapq.heappush(queue, (-float(affinity), int(rank[j]), j)); queued.add(j)
        cores.append(np.asarray(sorted(core), dtype=np.int64))
    if merge_small:
        owner = np.empty(graph.n, dtype=np.int64)
        for i, core in enumerate(cores):
            owner[core] = i
        # Each merge preserves connectivity because at least one true edge
        # crosses the two cores. Recompute adjacency after each owner update.
        changed = True
        while changed:
            changed = False
            for i in sorted(range(len(cores)), key=lambda j: (len(cores[j]), j)):
                core = cores[i]
                if not 0 < len(core) < min_core:
                    continue
                neighbors = graph.adjacency[core].indices
                ids, counts = np.unique(owner[neighbors], return_counts=True)
                eligible = [(int(count), -len(cores[j]), -int(rank[cores[j]].min()), int(j))
                            for j, count in zip(ids, counts)
                            if j != i and len(core) + len(cores[j]) <= max_core]
                if not eligible:
                    continue
                j = max(eligible)[-1]
                cores[j] = np.sort(np.concatenate([cores[j], core]))
                owner[core] = j; cores[i] = np.empty(0, dtype=np.int64)
                changed = True
    cores = sorted((c for c in cores if len(c)), key=lambda c: int(rank[c].min()))
    partitions = [complete_halo(graph, core, hops) for core in cores]
    validate_partition(graph, partitions, hops)
    return partitions

