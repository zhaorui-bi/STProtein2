"""Complete-dependency cost refinement must remain the same prediction model."""
from dataclasses import asdict

import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import connected_components

from stprotein2.graph import SpatialGraph, physical_graph, digest_array, validate_partition
from stprotein2.execution.planner import Budget, EmpiricalMemoryCost, PartitionPlan, complete_halo_sparse, plan_partition
from stprotein2.execution.refinement import refine_partition, plan_for_model
from test_architecture import setup_model


def ragged_graph():
    xy = np.array([(r, c) for r in range(12) for c in range(12)], dtype=float)
    graph = physical_graph(xy, k=4, radius=1.1)
    owner = (xy[:, 1] >= np.where(xy[:, 0] % 2 == 0, 4, 8)).astype(np.int64)
    parts = [complete_halo_sparse(graph, np.flatnonzero(owner == i), 1) for i in range(2)]
    report = {'required_hops': 1, 'graph_sha256': graph.fingerprint,
              'budget': asdict(Budget(max_nodes=144, max_edges=700)), 'memory_cost': None,
              'min_core': 16, 'max_core': 120, 'method': 'ragged_fixture',
              'owner_sha256': digest_array(owner), 'elapsed_seconds': 0.}
    return graph, PartitionPlan(parts, owner, report)


def contract(graph, result):
    assert validate_partition(graph, result.parts, result.report['required_hops'])
    for p in result.parts:
        assert connected_components(graph.adjacency[p.core][:, p.core], directed=False)[0] == 1
        assert len(p.nodes) <= result.report['budget']['max_nodes']
        assert np.all(result.owner[p.core] == result.owner[p.core[0]])


def test_ragged_boundaries_reduce_actual_halo_cost_without_removing_edges():
    graph, initial = ragged_graph()
    edge_before = graph.edge_index.copy(); owner_before = initial.owner.copy()
    result = refine_partition(graph, initial, max_passes=4, max_evaluations=2000)
    detail = result.report['refinement']
    assert detail['accepted_moves'] > 0
    assert detail['objective_after'] < detail['objective_before']
    assert sum(len(p.nodes) for p in result.parts) < sum(len(p.nodes) for p in initial.parts)
    assert all(h['after'] < h['before'] for h in detail['history'])
    assert len(result.parts) == len(initial.parts)
    np.testing.assert_array_equal(graph.edge_index, edge_before)
    np.testing.assert_array_equal(initial.owner, owner_before)
    contract(graph, result)


def test_cost_refinement_preserves_nonzero_model_predictions_and_gradients():
    model, args, _ = setup_model('cpu', checkpoint=True)
    graph = physical_graph(np.array([(r, c) for r in range(5) for c in range(6)], dtype=float), k=4, radius=1.5)
    result = plan_for_model(model, graph, method='grid', target_core=6, min_core=3, max_core=8,
                            budget=Budget(max_nodes=20), refinement={'max_evaluations': 100})
    contract(graph, result)
    model.train(); target = torch.randn(30, 5)
    full = model(*args)
    assert full['local_delta'].abs().max() > 1e-4
    (full['prediction'] - target).square().mean().backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    tiled = model(*args, partitions=[p.to('cpu') for p in reversed(result.parts)])
    torch.testing.assert_close(full['prediction'], tiled['prediction'], rtol=2e-5, atol=2e-6)
    (tiled['prediction'] - target).square().mean().backward()
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], rtol=2e-4, atol=5e-6)


def test_refinement_and_rna_ties_are_invariant_to_barcode_preserving_row_permutation():
    graph, initial = ragged_graph()
    keys = np.array([f'b_{i:04d}' for i in range(graph.n)])
    rng = np.random.default_rng(93); z = rng.normal(size=(graph.n, 4))
    settings = dict(max_passes=2, max_evaluations=400)
    a = refine_partition(graph, initial, node_keys=keys, embedding=z, **settings)
    permutation = rng.permutation(graph.n); inverse = np.argsort(permutation)
    edge = inverse[graph.edge_index]; edge = edge[:, np.lexsort((edge[1], edge[0]))]
    reordered = SpatialGraph(graph.n, edge, graph.coordinates[permutation], graph.distance_scale)
    parts = [complete_halo_sparse(reordered, np.sort(inverse[p.core]), 1) for p in initial.parts]
    owner = initial.owner[permutation]
    metadata = {**initial.report, 'graph_sha256': reordered.fingerprint, 'owner_sha256': digest_array(owner)}
    b = refine_partition(reordered, PartitionPlan(parts, owner, metadata), node_keys=keys[permutation],
                         embedding=z[permutation], **settings)
    np.testing.assert_array_equal(a.owner, b.owner[inverse])
    assert a.report['refinement']['objective_after'] == b.report['refinement']['objective_after']


def test_limits_profiles_and_tampered_owner_are_rejected_or_preserved():
    graph, initial = ragged_graph()
    unchanged = refine_partition(graph, initial, max_evaluations=0)
    np.testing.assert_array_equal(unchanged.owner, initial.owner)
    assert unchanged.report['refinement']['accepted_moves'] == 0
    limited = refine_partition(graph, initial, max_evaluations=1)
    assert limited.report['refinement']['evaluated_moves'] <= 1
    with pytest.raises(ValueError, match='integer search limits'):
        refine_partition(graph, initial, edge_weight=-1)
    bad = PartitionPlan(initial.parts, initial.owner[::-1], initial.report)
    with pytest.raises(ValueError, match='owner array'):
        refine_partition(graph, bad)
    cost = EmpiricalMemoryCost(1000, 20, 10, 'synthetic_unit_test_profile', safety_factor=1.)
    plan = plan_partition(graph, 'grid', target_core=30, min_core=8, max_core=60,
        budget=Budget(max_nodes=100, estimated_bytes=5500), memory_cost=cost)
    with pytest.raises(ValueError, match='same profiled memory'):
        refine_partition(graph, plan)
    result = refine_partition(graph, plan, memory_cost=cost, max_evaluations=100)
    assert result.report['estimated_peak_bytes'] <= 5500
    assert result.report['refinement']['peak_term_active']
    assert result.report['actual_memory_guaranteed'] is False
    contract(graph, result)


def test_disconnected_regions_and_isolated_nodes_remain_separate():
    coordinates = np.array([(i, j) for i in range(6) for j in range(6)] + [(100, 100)], dtype=float)
    graph = physical_graph(coordinates, k=4, radius=1.1)
    initial = plan_partition(graph, 'grid', target_core=10, min_core=3, max_core=16, budget=Budget(max_nodes=32))
    result = refine_partition(graph, initial, max_evaluations=100)
    contract(graph, result)
    isolated = next(p for p in result.parts if 36 in p.core)
    assert isolated.core.tolist() == isolated.nodes.tolist() == [36]
    one = physical_graph(np.array([[0., 0.]]))
    plan = plan_partition(one, 'grid', target_core=1, min_core=1, max_core=1)
    final = refine_partition(one, plan)
    assert final.report['edge_amplification'] == 0
    assert final.report['refinement']['accepted_moves'] == 0
