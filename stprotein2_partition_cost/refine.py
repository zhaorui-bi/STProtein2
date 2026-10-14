"""Deterministic boundary moves scored by complete Halo costs, not cut edges.

This first AH2 implementation keeps the initial number of Cores fixed. It does
not implement the proposed merge/split search, CUDA profiling or offloading.
Budget feasibility and exact model dependencies take priority over RNA ties.
"""
from dataclasses import asdict
import math
import time

import numpy as np
from scipy.sparse.csgraph import connected_components

from stprotein2.graph import digest_array, validate_partition
from stprotein2_partition.planner import (Budget, PartitionPlan, complete_halo_sparse,
                                         plan_for_model as initial_plan_for_model)


def refine_partition(graph, initial, *, node_keys=None, embedding=None, max_passes=2,
                     max_evaluations=2048, edge_weight=0.25, peak_weight=1.0, memory_cost=None):
    started = time.monotonic()
    if (not isinstance(max_passes, int) or max_passes < 0 or not isinstance(max_evaluations, int)
            or max_evaluations < 0 or any(not math.isfinite(v) or v < 0 for v in (edge_weight, peak_weight))):
        raise ValueError('Nonnegative finite costs and integer search limits required')
    if graph.n < 1 or (graph.adjacency != graph.adjacency.T).nnz:
        raise ValueError('Nonempty undirected physical graph required')
    keys = np.asarray(node_keys if node_keys is not None else np.arange(graph.n).astype(str))
    if keys.shape != (graph.n,) or len(np.unique(keys)) != graph.n:
        raise ValueError('Unique stable node identities required')
    order = np.argsort(keys, kind='stable')
    rank = np.empty(graph.n, dtype=np.int64); rank[order] = np.arange(graph.n)
    if embedding is not None:
        embedding = np.asarray(embedding, dtype=np.float64)
        if embedding.ndim != 2 or len(embedding) != graph.n or not np.isfinite(embedding).all():
            raise ValueError('Finite row-aligned routing embedding required')
    report = initial.report
    hops = report['required_hops']
    if report['graph_sha256'] != graph.fingerprint:
        raise ValueError('Initial plan belongs to a different graph')
    budget = Budget(**report['budget'])
    if report.get('memory_cost') != (asdict(memory_cost) if memory_cost else None):
        raise ValueError('Use the same profiled memory cost as the initial planner')
    if budget.estimated_bytes is not None and memory_cost is None:
        raise ValueError('A memory profile is required for an estimated byte budget')
    validate_partition(graph, initial.parts, hops)
    parts = sorted(initial.parts, key=lambda p: int(rank[p.core].min()))
    cores = [np.sort(p.core).copy() for p in parts]
    owner = np.full(graph.n, -1, dtype=np.int64)

    def costs(part):
        return (len(part.nodes), part.edge_index.shape[1], memory_cost.estimate(part) if memory_cost else 0)

    def feasible(values):
        return ((budget.max_nodes is None or values[0] <= budget.max_nodes)
                and (budget.max_edges is None or values[1] <= budget.max_edges)
                and (budget.estimated_bytes is None or values[2] <= budget.estimated_bytes))

    for i, core in enumerate(cores):
        if connected_components(graph.adjacency[core][:, core], directed=False)[0] != 1:
            raise ValueError('Initial Core is disconnected')
        if len(core) > report['max_core'] or not feasible(costs(parts[i])):
            raise ValueError('Initial partition is not feasible under its declared bounds')
        owner[core] = i
    if not np.array_equal(owner, initial.owner):
        raise ValueError('Initial owner array is inconsistent with canonical Cores')
    minimum = [min(len(c), report['min_core']) for c in cores]
    sizes = np.array([costs(p) for p in parts], dtype=np.int64)
    edge_denominator = max(1, graph.edge_index.shape[1])
    # Without a byte budget, a supplied empirical profile remains descriptive;
    # it cannot silently introduce a new arbitrary unit into the objective.
    use_peak = memory_cost is not None and budget.estimated_bytes is not None

    def objective(values):
        return (float(values[:, 0].sum()) / graph.n
                + edge_weight * float(values[:, 1].sum()) / edge_denominator
                + (peak_weight * float(values[:, 2].max()) / budget.estimated_bytes if use_peak else 0.))

    def route_sum(core):
        return embedding[core[np.argsort(rank[core], kind='stable')]].sum(0)

    sums = [route_sum(core) for core in cores] if embedding is not None else None
    current = original = objective(sizes)
    history = []; evaluated = disconnected = over_budget = 0
    completed_passes = 0
    for pass_id in range(max_passes):
        accepted_in_pass = 0
        for node in order:
            if evaluated >= max_evaluations:
                break
            node = int(node); source = int(owner[node]); source_core = cores[source]
            if len(source_core) <= minimum[source]:
                continue
            neighbors = graph.adjacency.indices[graph.adjacency.indptr[node]:graph.adjacency.indptr[node + 1]]
            recipients = sorted(set(owner[neighbors]) - {source}, key=lambda j: int(rank[cores[j]].min()))
            if not recipients:
                continue
            reduced = source_core[source_core != node]
            if connected_components(graph.adjacency[reduced][:, reduced], directed=False)[0] != 1:
                disconnected += 1
                continue
            source_part = complete_halo_sparse(graph, reduced, hops)
            source_cost = costs(source_part)
            if not feasible(source_cost):
                raise RuntimeError('Removing a node cannot expand its mandatory Halo')
            best = None
            for recipient in recipients:
                if evaluated >= max_evaluations:
                    break
                recipient = int(recipient)
                if len(cores[recipient]) >= report['max_core']:
                    continue
                evaluated += 1
                expanded = np.sort(np.append(cores[recipient], node))
                recipient_part = complete_halo_sparse(graph, expanded, hops)
                recipient_cost = costs(recipient_part)
                if not feasible(recipient_cost):
                    over_budget += 1
                    continue
                changed = sizes.copy(); changed[source] = source_cost; changed[recipient] = recipient_cost
                candidate = objective(changed)
                if candidate >= current - 1e-12:
                    continue
                route_change = 0.
                if sums is not None:
                    x = embedding[node]; a, b = sums[source], sums[recipient]
                    route_change = (a @ a / len(source_core) + b @ b / len(cores[recipient])
                                    - (a - x) @ (a - x) / len(reduced)
                                    - (b + x) @ (b + x) / len(expanded))
                key = (round(candidate, 12), round(float(route_change), 12), int(rank[cores[recipient]].min()))
                if best is None or key < best[0]:
                    best = (key, recipient, expanded, recipient_part, changed, candidate)
            if best is None:
                continue
            _, recipient, expanded, recipient_part, sizes, candidate = best
            if sums is not None:
                sums[source] = route_sum(reduced); sums[recipient] = route_sum(expanded)
            cores[source] = reduced; cores[recipient] = expanded
            parts[source] = source_part; parts[recipient] = recipient_part
            owner[node] = recipient
            history.append({'pass': pass_id, 'node_key': str(keys[node]), 'from_core': source,
                            'to_core': recipient, 'before': current, 'after': candidate})
            current = candidate; accepted_in_pass += 1
        completed_passes += 1
        if accepted_in_pass == 0 or evaluated >= max_evaluations:
            break
    # Rebuild rather than trusting the incremental cache before exposing output.
    parts = [complete_halo_sparse(graph, c, hops) for c in sorted(cores, key=lambda c: int(rank[c].min()))]
    owner.fill(-1)
    for i, part in enumerate(parts):
        owner[part.core] = i
        if connected_components(graph.adjacency[part.core][:, part.core], directed=False)[0] != 1:
            raise RuntimeError('Refinement disconnected a Core')
        if not feasible(costs(part)):
            raise RuntimeError('Final dependency rebuild violates the budget')
    validate_partition(graph, parts, hops)
    final = objective(np.array([costs(p) for p in parts], dtype=np.int64))
    if abs(final - current) > 1e-10 or final > original + 1e-12:
        raise RuntimeError('Cost bookkeeping or monotone refinement failed')
    row, col = graph.edge_index
    result = dict(report)
    result.update(method='cost_refined_' + report['method'], owner_sha256=digest_array(owner),
        cores=len(parts), core_sizes=[len(p.core) for p in parts], nodes_per_part=[len(p.nodes) for p in parts],
        edges_per_part=[p.edge_index.shape[1] for p in parts],
        halo_amplification=sum(len(p.nodes) for p in parts) / graph.n,
        edge_amplification=sum(p.edge_index.shape[1] for p in parts) / len(row) if len(row) else 0.,
        cut_edge_fraction=float(np.mean(owner[row] != owner[col])) if len(row) else 0.,
        estimated_peak_bytes=max(memory_cost.estimate(p) for p in parts) if memory_cost else None,
        elapsed_seconds=report['elapsed_seconds'] + time.monotonic() - started,
        refinement={'initial_method': report['method'], 'initial_owner_sha256': report['owner_sha256'],
            'max_passes': max_passes, 'max_evaluations': max_evaluations, 'completed_passes': completed_passes,
            'evaluated_moves': evaluated, 'accepted_moves': len(history), 'disconnected_donor_rejections': disconnected,
            'budget_rejections': over_budget, 'objective_before': original, 'objective_after': final,
            'edge_weight': edge_weight, 'peak_weight': peak_weight, 'peak_term_active': use_peak,
            'routing_tie_break_sha256': digest_array(embedding) if embedding is not None else None,
            'search_exhaustive': False, 'core_count_preserved': True,
            'boundary_moves_only': True, 'history': history, 'elapsed_seconds': time.monotonic() - started})
    return PartitionPlan(parts, owner, result)


def plan_for_model(model, graph, *, refinement=None, **kwargs):
    initial = initial_plan_for_model(model, graph, **kwargs)
    return refine_partition(graph, initial, node_keys=kwargs.get('node_keys'), embedding=kwargs.get('embedding'),
                            memory_cost=kwargs.get('memory_cost'), **(refinement or {}))
