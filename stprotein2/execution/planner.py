"""Connected partitions with exact Halo node/edge limits and explicit estimates.

Local dependency extraction is sparse; there is no per-block full-N membership
array or scan of every full-graph edge. Full-graph sheaf coefficients and their
gradients remain the responsibility of the unchanged model.
"""
from collections import deque
from dataclasses import asdict, dataclass
import heapq
import importlib
import importlib.metadata
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from stprotein2.graph import CoreHalo, digest_array


@dataclass(frozen=True)
class EmpiricalMemoryCost:
    fixed_bytes: float
    bytes_per_node: float
    bytes_per_edge: float
    profile_id: str
    safety_factor: float = 1.2

    def __post_init__(self):
        if (not all(math.isfinite(x) and x >= 0 for x in
                    (self.fixed_bytes, self.bytes_per_node, self.bytes_per_edge))
                or not math.isfinite(self.safety_factor) or self.safety_factor < 1 or not self.profile_id):
            raise ValueError('Nonnegative finite costs, a profile ID and safety factor >= 1 required')

    def estimate(self, part):
        return math.ceil(self.safety_factor * (self.fixed_bytes + self.bytes_per_node * len(part.nodes)
                                               + self.bytes_per_edge * part.edge_index.shape[1]))


@dataclass(frozen=True)
class Budget:
    max_nodes: int | None = 1024
    max_edges: int | None = None
    estimated_bytes: int | None = None

    def __post_init__(self):
        if any(value is not None and (not isinstance(value,int) or value < 1)
               for value in (self.max_nodes,self.max_edges,self.estimated_bytes)):
            raise ValueError('Partition limits must be positive integers or None')


class PartitionBudgetError(RuntimeError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


@dataclass
class PartitionPlan:
    parts: list
    owner: np.ndarray
    report: dict


def _canonical_rank(graph, node_keys):
    keys = np.asarray(node_keys if node_keys is not None else np.arange(graph.n).astype(str))
    if keys.shape != (graph.n,) or len(np.unique(keys)) != graph.n:
        raise ValueError('Unique node identities required for deterministic tie breaking')
    order = np.argsort(keys,kind='stable')
    rank = np.empty(graph.n,dtype=np.int64);rank[order]=np.arange(graph.n)
    return order,rank


def induced_edges(graph, nodes):
    """Build the induced local COO from only the selected CSR rows."""
    nodes=np.asarray(nodes,dtype=np.int64)
    if len(nodes)==0:return np.empty((2,0),dtype=np.int64)
    indptr,indices=graph.adjacency.indptr,graph.adjacency.indices
    lengths=indptr[nodes+1]-indptr[nodes]
    total=int(lengths.sum())
    if total==0:return np.empty((2,0),dtype=np.int64)
    starts=np.repeat(indptr[nodes],lengths)
    offsets=np.arange(total)-np.repeat(np.cumsum(lengths)-lengths,lengths)
    neighbors=indices[starts+offsets]
    cols=np.searchsorted(nodes,neighbors)
    valid=cols<len(nodes)
    valid[valid] &= nodes[cols[valid]]==neighbors[valid]
    rows=np.repeat(np.arange(len(nodes)),lengths)
    edges=np.stack([rows[valid],cols[valid]])
    return edges[:,np.lexsort((edges[1],edges[0]))]


def complete_halo_sparse(graph, core, hops=2):
    core=np.asarray(core,dtype=np.int64)
    if (not len(core) or len(np.unique(core))!=len(core) or core.min()<0 or core.max()>=graph.n
            or not isinstance(hops,int) or hops<0):
        raise ValueError('Invalid Core or dependency depth')
    nodes=np.sort(core);frontier=nodes
    for _ in range(hops):
        neighbors=np.unique(graph.adjacency[frontier].indices)
        frontier=np.setdiff1d(neighbors,nodes,assume_unique=True)
        nodes=np.union1d(nodes,frontier)
        if not len(frontier):break
    return CoreHalo(core,nodes,np.searchsorted(nodes,core),induced_edges(graph,nodes),hops)


def _split_connected(graph, core, rank):
    """Two geodesic seeds; multi-source BFS preserves connected children."""
    core=np.sort(core)
    if len(core)<2:raise ValueError('A singleton Core cannot be split')
    edges=induced_edges(graph,core)
    adj=sparse.csr_matrix((np.ones(edges.shape[1]),edges),shape=(len(core),len(core)))
    local_rank=rank[core]
    def farthest(seed):
        distance=np.full(len(core),-1,dtype=np.int64);distance[seed]=0
        queue=deque([seed])
        while queue:
            i=queue.popleft()
            for j in adj.indices[adj.indptr[i]:adj.indptr[i+1]]:
                if distance[j]<0:distance[j]=distance[i]+1;queue.append(int(j))
        if (distance<0).any():raise ValueError('Bisection requires a connected Core')
        candidates=np.flatnonzero(distance==distance.max())
        return int(candidates[np.argmin(local_rank[candidates])])
    first=farthest(int(np.argmin(local_rank)));second=farthest(first)
    owner=np.full(len(core),-1,dtype=np.int8);owner[first]=0;owner[second]=1
    queue=deque([first,second])
    while queue:
        i=queue.popleft()
        adjacent=adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        for j in adjacent[np.argsort(local_rank[adjacent],kind='stable')]:
            if owner[j]<0:owner[j]=owner[i];queue.append(int(j))
    return [core[owner==0],core[owner==1]]


def _connected_cores(graph, raw_labels, rank):
    labels=np.asarray(raw_labels)
    if labels.shape!=(graph.n,):raise ValueError('Owner labels must match the graph')
    row,col=graph.edge_index
    retain=labels[row]==labels[col]
    pruned=sparse.csr_matrix((np.ones(int(retain.sum())),(row[retain],col[retain])),shape=(graph.n,graph.n))
    count,component=connected_components(pruned,directed=False)
    order=np.argsort(component,kind='stable')
    boundaries=np.flatnonzero(np.diff(component[order]))+1
    cores=[np.sort(c) for c in np.split(order,boundaries)]
    return sorted(cores,key=lambda c:int(rank[c].min()))


def _merge_small(graph, cores, rank, minimum, maximum):
    owner=np.empty(graph.n,dtype=np.int64)
    for i,c in enumerate(cores):owner[c]=i
    changed=True
    while changed:
        changed=False
        for i in sorted(range(len(cores)),key=lambda k:(len(cores[k]),int(rank[cores[k]].min()) if len(cores[k]) else -1)):
            core=cores[i]
            if not 0<len(core)<minimum:continue
            candidates,counts=np.unique(owner[graph.adjacency[core].indices],return_counts=True)
            choices=[(int(count),-len(cores[j]),-int(rank[cores[j]].min()),int(j))
                     for j,count in zip(candidates,counts)
                     if j!=i and len(cores[j]) and len(core)+len(cores[j])<=maximum]
            if choices:
                j=max(choices)[-1]
                cores[j]=np.sort(np.concatenate([cores[j],core]));owner[core]=j
                cores[i]=np.empty(0,dtype=np.int64);changed=True
    return [c for c in cores if len(c)]


def _adaptive_labels(graph, order, rank, target, minimum, maximum, score, embedding):
    # Preserve the frozen adaptive route semantics while extracting only owner
    # labels; the complete Halo is constructed once by the sparse planner.
    assigned=np.zeros(graph.n,dtype=bool);distance=np.full(graph.n,np.inf)
    labels=np.full(graph.n,-1,dtype=np.int64);label=0
    while not assigned.all():
        candidates=order[~assigned[order]]
        seed=int(candidates[np.argmax(distance[candidates])]);distance[seed]=0
        frontier=deque([seed])
        while frontier:
            i=frontier.popleft()
            for j in graph.adjacency.indices[graph.adjacency.indptr[i]:graph.adjacency.indptr[i+1]]:
                if distance[i]+1<distance[j]:distance[j]=distance[i]+1;frontier.append(int(j))
        core=[];queue=[(0.,int(rank[seed]),seed)];queued={seed};limit=target
        while queue and len(core)<limit:
            _,_,i=heapq.heappop(queue)
            if assigned[i]:continue
            assigned[i]=True;labels[i]=label;core.append(i)
            if score is not None:
                quantum=min(32,max(1,target//8))
                limit=int(np.clip(quantum*round((2*target/(1+3*float(np.mean(score[core]))))/quantum),minimum,maximum))
            for j in graph.adjacency.indices[graph.adjacency.indptr[i]:graph.adjacency.indptr[i+1]]:
                j=int(j)
                if assigned[j] or j in queued:continue
                d=np.linalg.norm(graph.coordinates[i]-graph.coordinates[j])/graph.distance_scale
                molecular=float(np.sum((embedding[i]-embedding[j])**2)) if embedding is not None else 0.
                heapq.heappush(queue,(-float(np.exp(-d*d-molecular)),int(rank[j]),j));queued.add(j)
        label+=1
    return labels


def _optional(name, dependency_path):
    if dependency_path is not None:
        path=str(Path(dependency_path).resolve())
        if path not in sys.path:sys.path.insert(0,path)
    try:return importlib.import_module(name)
    except ImportError as exc:raise ImportError(f'{name} is required for this partitioner; install the pinned optional dependencies') from exc


def initial_labels(graph, method, order, rank, target, minimum, maximum, score, embedding,
                   seed, grid_width, resolution, dependency_path):
    metadata={'algorithm':method}
    if method=='adaptive':
        metadata.update(size_rule='clip(round_quantum(2*target/(1+3*mean_score)),min,max)',
                        rounding_quantum=min(32,max(1,target//8)))
        return _adaptive_labels(graph,order,rank,target,minimum,maximum,score,embedding),metadata
    if method=='grid':
        width=float(grid_width if grid_width is not None else math.sqrt(target)*graph.distance_scale)
        if not math.isfinite(width) or width<=0:raise ValueError('Positive finite grid width required')
        cells=np.floor((graph.coordinates-graph.coordinates.min(0))/width).astype(np.int64)
        _,labels=np.unique(cells,axis=0,return_inverse=True)
        metadata.update(grid_width=width,coordinate_origin=graph.coordinates.min(0).tolist())
        return labels,metadata
    if method=='metis':
        native=_optional('pymetis',dependency_path)
        # Canonical row order makes a row permutation with stable barcodes
        # independent of native-library vertex numbering.
        adj=graph.adjacency[order][:,order].tocsr();adj.sort_indices()
        dtype=native.zero_copy_dtype()
        csr=native.CSRAdjacency(adj.indptr.astype(dtype),adj.indices.astype(dtype))
        parts=min(graph.n,max(1,math.ceil(graph.n/target)))
        result=native.part_graph(parts,adjacency=csr,options=native.Options(seed=seed,ufactor=30))
        labels=np.empty(graph.n,dtype=np.int64);labels[order]=result.vertex_part
        metadata.update(version=importlib.metadata.version('pymetis'),requested_parts=parts,
                        native_edge_cuts=int(result.edge_cuts),seed=seed,ufactor=30)
        return labels,metadata
    if method=='leiden':
        if not math.isfinite(resolution) or resolution<=0:raise ValueError('Positive Leiden resolution required')
        ig=_optional('igraph',dependency_path);la=_optional('leidenalg',dependency_path)
        row,col=graph.edge_index
        edges=np.column_stack([rank[row[row<col]],rank[col[row<col]]])
        edges=np.sort(edges,axis=1);edges=edges[np.lexsort((edges[:,1],edges[:,0]))]
        native=ig.Graph(n=graph.n,edges=edges,directed=False)
        partition=la.find_partition(native,la.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,seed=seed,n_iterations=2)
        labels=np.empty(graph.n,dtype=np.int64);labels[order]=partition.membership
        metadata.update(version=importlib.metadata.version('leidenalg'),igraph_version=ig.__version__,
                        resolution=resolution,seed=seed,iterations=2,weights='unweighted_physical_edges')
        return labels,metadata
    raise ValueError('Unknown owner partition algorithm')


def plan_partition(graph, method='adaptive', *, target_core=256, min_core=128, max_core=512,
                   required_hops=2, node_keys=None, score=None, embedding=None, budget=Budget(),
                   memory_cost=None, seed=2025, grid_width=None, resolution=1., dependency_path=None):
    """Return model-compatible complete-Halo tuples without changing graph support.

    Node and physical-edge caps are exact. Byte limits are empirical estimates,
    never a guarantee of GPU fit. Full-slice memory is not removed by this API.
    """
    start=time.monotonic()
    if (graph.n<1 or np.shape(graph.coordinates)!=(graph.n,2) or not np.isfinite(graph.coordinates).all()
            or not math.isfinite(graph.distance_scale) or graph.distance_scale<=0):
        raise ValueError('Nonempty graph with finite coordinates and positive distance scale required')
    if not 1<=min_core<=target_core<=max_core:raise ValueError('Invalid Core size bounds')
    if not isinstance(required_hops,int) or required_hops<0:raise ValueError('Invalid dependency depth')
    if (graph.adjacency!=graph.adjacency.T).nnz:raise ValueError('Owner planning requires an undirected physical graph')
    if score is not None and (np.shape(score)!=(graph.n,) or not np.isfinite(score).all() or np.any(np.asarray(score)<0) or np.any(np.asarray(score)>1)):
        raise ValueError('Adaptive score must be finite and in [0,1]')
    if embedding is not None and (np.ndim(embedding)!=2 or len(embedding)!=graph.n or not np.isfinite(embedding).all()):
        raise ValueError('Invalid routing embedding')
    if budget.estimated_bytes is not None and memory_cost is None:
        raise ValueError('A profiled memory cost is required for an estimated byte budget')
    order,rank=_canonical_rank(graph,node_keys)
    raw,metadata=initial_labels(graph,method,order,rank,target_core,min_core,max_core,score,embedding,
                               seed,grid_width,resolution,dependency_path)
    raw_count=len(np.unique(raw))
    cores=_connected_cores(graph,raw,rank);connected_count=len(cores)
    pending=deque(cores);cores=[]
    while pending:
        core=pending.popleft()
        if len(core)>max_core:pending.extend(_split_connected(graph,core,rank))
        else:cores.append(core)
    cores=_merge_small(graph,cores,rank,min_core,max_core)
    pending=deque(complete_halo_sparse(graph,c,required_hops) for c in cores)
    parts=[];splits=0;neutral_splits=0
    def cost(part):
        return {'nodes':len(part.nodes),'edges':int(part.edge_index.shape[1]),
                'estimated_bytes':memory_cost.estimate(part) if memory_cost is not None else None}
    def exceeds(values):
        return ((budget.max_nodes is not None and values['nodes']>budget.max_nodes)
                or (budget.max_edges is not None and values['edges']>budget.max_edges)
                or (budget.estimated_bytes is not None and values['estimated_bytes']>budget.estimated_bytes))
    while pending:
        part=pending.popleft();core=part.core;current=cost(part)
        if not exceeds(current):parts.append(part);continue
        if len(core)==1:
            raise PartitionBudgetError('A singleton mandatory Halo exceeds the declared limits; dependencies were not truncated',
                {'node':int(core[0]),'mandatory_cost':current,'budget':asdict(budget),
                 'estimated_memory_only':budget.estimated_bytes is not None})
        children=_split_connected(graph,core,rank)
        # A neutral first split is valid: cycles can require another split
        # before either child's Halo becomes smaller. Core sizes always shrink.
        child_parts=[complete_halo_sparse(graph,c,required_hops) for c in children]
        if max(len(p.nodes) for p in child_parts)==len(part.nodes):neutral_splits+=1
        pending.extend(child_parts);splits+=1
    parts.sort(key=lambda p:int(rank[p.core].min()))
    owner=np.full(graph.n,-1,dtype=np.int64)
    for i,part in enumerate(parts):
        if (owner[part.core]>=0).any():raise RuntimeError('Duplicate Core ownership')
        owner[part.core]=i
    if (owner<0).any():raise RuntimeError('Missing Core owner')
    row,col=graph.edge_index
    report={'method':method,'algorithm_metadata':metadata,'graph_sha256':graph.fingerprint,
            'coordinates_sha256':digest_array(graph.coordinates),'canonical_order_sha256':digest_array(order),
            'target_core':target_core,'min_core':min_core,'max_core':max_core,
            'routing_score_sha256':digest_array(score) if score is not None else None,
            'routing_embedding_sha256':digest_array(embedding) if embedding is not None else None,
            'owner_sha256':digest_array(owner),'required_hops':required_hops,'requires_full_graph_coefficients':True,
            'raw_groups':raw_count,'connected_groups_before_size_repair':connected_count,'cores':len(parts),
            'core_sizes':[len(p.core) for p in parts],'nodes_per_part':[len(p.nodes) for p in parts],
            'edges_per_part':[int(p.edge_index.shape[1]) for p in parts],
            'halo_amplification':sum(len(p.nodes) for p in parts)/graph.n,
            'edge_amplification':sum(p.edge_index.shape[1] for p in parts)/len(row) if len(row) else 0.,
            'cut_edge_fraction':float(np.mean(owner[row]!=owner[col])) if len(row) else 0.,
            'budget':asdict(budget),'budget_splits':splits,'neutral_halo_splits':neutral_splits,
            'memory_cost':asdict(memory_cost) if memory_cost else None,'actual_memory_guaranteed':False,
            'estimated_peak_bytes':max(memory_cost.estimate(p) for p in parts) if memory_cost else None,
            'full_slice_encoding_and_coefficients_remain':True,'elapsed_seconds':time.monotonic()-start}
    return PartitionPlan(parts,owner,report)


def plan_for_model(model, graph, **kwargs):
    """Only the currently audited one-layer, two-propagation CSNN is supported."""
    from stprotein2.models.residual import STProtein2
    from stprotein2.models.anchor import AnchoredSTProtein2
    if type(model) not in (STProtein2,AnchoredSTProtein2) or model.cfg.local_operator!='csnn':
        raise ValueError('Derive a dependency specification before partitioning a different model/operator')
    if 'required_hops' in kwargs:raise ValueError('Model dependency depth cannot be overridden')
    return plan_partition(graph,required_hops=2,**kwargs)
