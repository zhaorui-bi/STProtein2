import copy

import joblib
import numpy as np
import pytest
import torch

from stprotein2.config import ModelConfig
from stprotein2.data import RNAView
from stprotein2.graph import physical_graph
from stprotein2.model import STProtein2
from stprotein2.prototypes import EMATeacher
from stprotein2_balanced_initialization import sample_references, fit_router, initialize_prototypes


def view(name,group,n,seed=1):
    rng=np.random.default_rng(seed)
    return RNAView(name,group,rng.normal(size=(n,6)).astype(np.float32),np.ones(6,np.float32),
        rng.normal(size=(n,4)).astype(np.float32),np.column_stack([np.arange(n),np.zeros(n)]).astype(float),
        np.asarray([f"{name}_{i:04d}" for i in range(n)]),tuple(f"g{i}" for i in range(6)))


def reversed_view(v):
    r=copy.deepcopy(v)
    for key in ("x","qc","coordinates","barcodes"):
        setattr(r,key,getattr(r,key)[::-1].copy())
    return r


def test_capacity_capped_source_slice_quotas_and_row_order_invariance():
    values=[view("large","A",1000),view("tiny","A",5),view("middle","B",51)]
    names=[v.name for v in values]
    first=sample_references(values,reference_names=names,per_source=200,seed=3)
    second=sample_references([reversed_view(v) for v in values[::-1]],reference_names=names[::-1],per_source=200,seed=3)
    assert first.record==second.record
    assert first.record["actual_per_source"]==51
    quotas={r["slice"]:r["quota"] for r in first.record["slices"]}
    assert quotas=={"large":46,"tiny":5,"middle":51}
    for v in values:
        indices=first.indices[v.name]
        assert len(np.unique(indices))==len(indices)
        assert v.barcodes[indices].tolist()==next(r["barcodes"] for r in first.record["slices"] if r["slice"]==v.name)
    even=[view("a","A",100),view("b","A",100),view("c","A",100),view("z","B",100)]
    p=sample_references(even,reference_names=[v.name for v in even],per_source=31,seed=7)
    q=[r["quota"] for r in p.record["slices"] if r["source"]=="A"]
    assert sum(q)==31 and max(q)-min(q)==1
    with pytest.raises(ValueError,match="allowed reference"):
        sample_references(values+[view("query","Q",10)],reference_names=names,per_source=10)
    with pytest.raises(ValueError):
        sample_references(values,reference_names=names,per_source=True)


def test_router_identical_under_input_permutation_and_serialization(tmp_path):
    values=[view("one","A",33),view("two","A",17,2),view("three","B",42,3)]
    names=[v.name for v in values]
    a,record=fit_router(values,reference_names=names,components=4,clusters=4,per_source=24,seed=13)
    b,other=fit_router([reversed_view(v) for v in values[::-1]],reference_names=names[::-1],components=4,clusters=4,per_source=24,seed=13)
    assert record==other and a.fitted_groups_==["A","B"]
    np.testing.assert_array_equal(a.pca_.components_,b.pca_.components_)
    np.testing.assert_array_equal(a.centers_,b.centers_)
    query=view("query","held",11,7)
    expected=a.transform(query.x)
    artifact=tmp_path/"router.joblib"
    joblib.dump(a,artifact)
    restored=joblib.load(artifact)
    for x,y in zip(expected,restored.transform(query.x)):
        np.testing.assert_array_equal(x,y)
    assert restored.sampling_record_==record
    assert "query" not in record["reference_names"]


def test_state_then_niche_initialization_teacher_slots_and_mode_restore():
    torch.manual_seed(19)
    cfg=ModelConfig(n_genes=6,n_proteins=5,width=12,hidden=8,members=2,
        state_prototypes=4,niche_prototypes=3,prototype_dim=6,experts=1,stalk=2,channels=2)
    model=STProtein2(cfg).train()
    before={k:v.clone() for k,v in model.state_dict().items()}
    values=[view("one","A",19),view("two","A",11,2),view("three","B",22,3)]
    names=[v.name for v in values]
    graphs={v.name:physical_graph(v.coordinates,k=2) for v in values}
    report=initialize_prototypes(model,values,graphs,reference_names=names,per_source=12,seed=5)
    assert model.training
    assert [r["bank"] for r in report["stages"]]==["state","niche"]
    assert report["stages"][1]["niche_uses_initialized_state"]
    assert all(r["unique_samples"]==24 for r in report["stages"])
    changed={k for k,v in model.state_dict().items() if not torch.equal(v,before[k])}
    assert changed=={"representations.state.codes","representations.niche.codes"}
    teacher=EMATeacher(model.representations)
    for k,v in model.representations.state_dict().items():
        assert torch.equal(v,teacher.representations.state_dict()[k])
    identical=STProtein2(cfg)
    identical.load_state_dict(before)
    again=initialize_prototypes(identical,values[::-1],graphs,reference_names=names[::-1],per_source=12,seed=5)
    assert report==again
    for k,v in model.state_dict().items():
        assert torch.equal(v,identical.state_dict()[k])
    previous={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError,match="Insufficient unique"):
        initialize_prototypes(model,values,graphs,reference_names=names,per_source=1)
    assert all(torch.equal(v,previous[k]) for k,v in model.state_dict().items())
