"""Mathematical and scientific acceptance checks for the expanded evaluation."""
import numpy as np
import pytest
from stprotein2.graph import SpatialGraph
from stprotein2.evaluation.spatial import compare, spatial_statistics, extended_metrics


def test_spatial_matches_dense_formula_and_affine_invariance():
    n=5
    edge=np.array([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]])
    graph=SpatialGraph(n,edge,np.column_stack([np.arange(n),np.zeros(n)]),1.)
    x=np.array([[0.,2.],[1.,1.],[3.,4.],[2.,8.],[7.,3.]])
    result=spatial_statistics(x,graph)
    w=graph.adjacency.toarray();z=x-x.mean(0)
    for j in range(2):
        expected_i=n/w.sum()*(z[:,j]@w@z[:,j])/(z[:,j]@z[:,j])
        differences=x[:,j,None]-x[None,:,j]
        expected_c=(n-1)/(2*w.sum())*np.sum(w*differences**2)/(z[:,j]@z[:,j])
        assert result['moran'][j]==pytest.approx(expected_i)
        assert result['geary'][j]==pytest.approx(expected_c)
    transformed=spatial_statistics(x*np.array([-2.,3.])+np.array([4.,-1.]),graph)
    np.testing.assert_allclose(result['moran'],transformed['moran'])
    np.testing.assert_allclose(result['geary'],transformed['geary'])
    assert spatial_statistics(np.ones((n,2)),graph)=={'moran':[None,None],'geary':[None,None]}


def test_perfect_prediction_agrees_and_residual_is_undefined():
    rng=np.random.default_rng(8);n=20
    edge=np.concatenate([np.stack([np.arange(n-1),np.arange(1,n)]),np.stack([np.arange(1,n),np.arange(n-1)])],1)
    graph=SpatialGraph(n,edge,np.column_stack([np.arange(n),np.zeros(n)]),1.)
    y=rng.normal(size=(n,5))
    score=extended_metrics(y,y,graph)['summary']
    assert score['moran_alignment_MAE']==0 and score['geary_alignment_RMSE']==0
    assert score['residual_moran_mean'] is None and score['undefined_residual_spatial_channels']==5
    for k in [4,6,8]:
        for key in ['ari','nmi','domain_homogeneity','domain_completeness']:
            assert score[f'cluster_k{k}_{key}']==pytest.approx(1.)


def test_all_cell_acceptance_rejects_missing_losing_and_undefined_panels():
    baseline=[{'query':q,'seed':s,'error':1.,'corr':.5} for q in ['a','b'] for s in [1,2,3]]
    candidate=[{**r,'error':.9,'corr':.6} for r in baseline]
    keys={'error':'lower','corr':'higher'}
    assert compare(baseline,candidate,['a','b'],[1,2,3],keys)['all_quality_cells_strictly_won']
    with pytest.raises(ValueError):compare(baseline,candidate[:-1],['a','b'],[1,2,3],keys)
    candidate[-1]['corr']=None
    candidate[0]['error']=2.
    result=compare(baseline,candidate,['a','b'],[1,2,3],keys)
    assert not result['all_quality_cells_strictly_won']
    assert result['counts']['undefined']==1 and result['counts']['loss']==1
