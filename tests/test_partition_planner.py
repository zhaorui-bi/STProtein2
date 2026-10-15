"""Connected native partition controls must preserve the actual model function."""
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
import torch

from stprotein2.graph import SpatialGraph, physical_graph, complete_halo, validate_partition
from stprotein2.execution.planner import Budget, EmpiricalMemoryCost, PartitionBudgetError, complete_halo_sparse, plan_partition, plan_for_model
from test_architecture import setup_model


DEPENDENCIES=Path(__file__).resolve().parents[1]/'external/partition_deps'


def require_native(method):
    if method in {'metis','leiden'} and not DEPENDENCIES.exists():
        pytest.skip('Install pinned optional partition dependencies to exercise native controls')


def assert_contract(graph,plan):
    assert validate_partition(graph,plan.parts,plan.report['required_hops'])
    for part in plan.parts:
        assert connected_components(graph.adjacency[part.core][:,part.core],directed=False)[0]==1
        expected=complete_halo(graph,part.core,plan.report['required_hops'])
        np.testing.assert_array_equal(part.nodes,expected.nodes)
        np.testing.assert_array_equal(part.edge_index,expected.edge_index)
        assert np.all(plan.owner[part.core]==plan.owner[part.core[0]])


def test_sparse_dependency_extraction_matches_frozen_reference_and_isolation():
    coords=np.array([(i,j) for i in range(8) for j in range(7)]+[(100,100)],dtype=float)
    graph=physical_graph(coords,k=4,radius=1.1)
    for hops in (0,1,2,3):
        for core in ([0,1,7],[25],[56],[1,40,56]):
            actual=complete_halo_sparse(graph,core,hops)
            expected=complete_halo(graph,core,hops)
            for field in ('core','nodes','core_local','edge_index'):
                np.testing.assert_array_equal(getattr(actual,field),getattr(expected,field))
    # A huge grid cell contains disconnected physical regions; they must split.
    plan=plan_partition(graph,'grid',target_core=20,min_core=5,max_core=60,grid_width=1000,budget=Budget(None))
    assert plan.report['raw_groups']==1
    assert plan.report['connected_groups_before_size_repair']==2
    assert_contract(graph,plan)
    isolated=[p for p in plan.parts if 56 in p.core][0]
    assert isolated.core.tolist()==isolated.nodes.tolist()==[56]


@pytest.mark.parametrize('method',['grid','adaptive','metis','leiden'])
def test_native_owner_controls_preserve_nonzero_outputs_and_all_gradients(method):
    require_native(method)
    model,args,_=setup_model('cpu',checkpoint=True)
    graph=physical_graph(np.array([(r,c) for r in range(5) for c in range(6)],dtype=float),k=4,radius=1.5)
    plan=plan_for_model(model,graph,method=method,target_core=6,min_core=3,max_core=8,
                       budget=Budget(max_nodes=20),dependency_path=DEPENDENCIES)
    assert_contract(graph,plan)
    assert max(map(lambda p:len(p.nodes),plan.parts))<=20
    target=torch.randn(len(args[0]),5)
    model.train()
    full=model(*args)
    assert full['local_delta'].abs().max()>1e-4
    (full['prediction']-target).square().mean().backward()
    expected={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    tiled=model(*args,partitions=[p.to('cpu') for p in reversed(plan.parts)])
    torch.testing.assert_close(tiled['prediction'],full['prediction'],rtol=2e-5,atol=2e-6)
    (tiled['prediction']-target).square().mean().backward()
    for name,p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad,expected[name],rtol=2e-4,atol=5e-6,msg=lambda m:name+': '+m)


@pytest.mark.parametrize('method',['grid','adaptive','metis','leiden'])
def test_owner_identity_is_invariant_to_node_order_given_stable_barcodes(method):
    require_native(method)
    rng=np.random.default_rng(83)
    coords=np.array([(i,j) for i in range(7) for j in range(6)],dtype=float)
    graph=physical_graph(coords,k=4,radius=1.1)
    keys=np.array([f'barcode_{i:04d}' for i in range(len(coords))])
    score=rng.uniform(0,1,len(coords));embedding=rng.normal(size=(len(coords),3))
    kwargs=dict(target_core=8,min_core=3,max_core=12,budget=Budget(max_nodes=30),dependency_path=DEPENDENCIES)
    a=plan_partition(graph,method,node_keys=keys,score=score,embedding=embedding,**kwargs)
    permutation=rng.permutation(graph.n);inverse=np.empty(graph.n,dtype=np.int64);inverse[permutation]=np.arange(graph.n)
    edge=inverse[graph.edge_index];edge=edge[:,np.lexsort((edge[1],edge[0]))]
    reordered=SpatialGraph(graph.n,edge,coords[permutation],graph.distance_scale)
    b=plan_partition(reordered,method,node_keys=keys[permutation],score=score[permutation],embedding=embedding[permutation],**kwargs)
    np.testing.assert_array_equal(a.owner,b.owner[inverse])


def test_budget_allows_neutral_intermediate_splits_and_rejects_singleton_closure():
    n=8;theta=np.arange(n)*2*np.pi/n
    pairs={(i,(i+1)%n) for i in range(n)}|{((i+1)%n,i) for i in range(n)}
    cycle=SpatialGraph(n,np.array(sorted(pairs),dtype=np.int64).T,np.column_stack([np.cos(theta),np.sin(theta)]),1.)
    plan=plan_partition(cycle,'grid',target_core=8,min_core=1,max_core=8,grid_width=100,budget=Budget(max_nodes=6))
    assert plan.report['neutral_halo_splits']>0
    assert max(len(p.nodes) for p in plan.parts)<=6
    assert_contract(cycle,plan)
    pairs={(0,i) for i in range(1,10)}|{(i,0) for i in range(1,10)}
    star=SpatialGraph(10,np.array(sorted(pairs),dtype=np.int64).T,np.column_stack([np.arange(10),np.zeros(10)]),1.)
    with pytest.raises(PartitionBudgetError,match='singleton mandatory Halo') as caught:
        plan_partition(star,'grid',target_core=10,min_core=1,max_core=10,grid_width=100,budget=Budget(max_nodes=9))
    assert caught.value.diagnostic['mandatory_cost']['nodes']==10


def test_byte_budget_requires_profile_and_does_not_hide_global_cost():
    coords=np.array([(i,0) for i in range(8)],dtype=float)
    graph=physical_graph(coords,k=2,radius=1.1)
    settings=dict(method='grid',target_core=4,min_core=1,max_core=8,grid_width=100)
    with pytest.raises(ValueError,match='profiled memory cost'):
        plan_partition(graph,budget=Budget(None,estimated_bytes=1000),**settings)
    cost=EmpiricalMemoryCost(1000,20,10,'synthetic_test_profile',safety_factor=1.)
    with pytest.raises(PartitionBudgetError):
        plan_partition(graph,budget=Budget(None,estimated_bytes=900),memory_cost=cost,**settings)
    plan=plan_partition(graph,budget=Budget(None,estimated_bytes=1200),memory_cost=cost,**settings)
    assert plan.report['estimated_peak_bytes']<=1200
    assert plan.report['actual_memory_guaranteed'] is False
    assert_contract(graph,plan)
    model,_,_=setup_model()
    with pytest.raises(ValueError,match='cannot be overridden'):
        plan_for_model(model,graph,required_hops=0)


def test_adaptive_requested_scale_changes_block_sizes():
    graph=physical_graph(np.array([(i,j) for i in range(30) for j in range(30)],dtype=float),k=4,radius=1.1)
    common=dict(min_core=16,max_core=512,score=np.zeros(graph.n),budget=Budget(None))
    small=plan_partition(graph,'adaptive',target_core=64,**common)
    large=plan_partition(graph,'adaptive',target_core=256,**common)
    assert len(small.parts)>len(large.parts)
    # Tiny unmergeable remainders can dominate an unweighted block median.
    # Compare the Core size seen by a uniformly sampled node instead.
    a=np.asarray(small.report['core_sizes']);b=np.asarray(large.report['core_sizes'])
    assert np.average(a,weights=a)<np.average(b,weights=b)
