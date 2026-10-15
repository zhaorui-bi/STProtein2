import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import torch

from stprotein2.config import ModelConfig, TrainConfig
from stprotein2.data import RNAView, ReferenceTransform, TrainingSlice, PANEL, count_thinning_views
from stprotein2.graph import FrozenRouter
from stprotein2.io import freeze_run
from research.joint.trainer import fit
from stprotein2.training.basic import predict


def raw_sample(seed, n=18, genes=12):
    rng=np.random.default_rng(seed)
    a=ad.AnnData(sparse.csr_matrix(rng.poisson(3,(n,genes)).astype(np.float32)))
    a.var_names=[f'g{i}' for i in range(genes)]
    a.obs_names=[f's{seed}_{i}' for i in range(n)]
    a.obsm['spatial']=np.array([(i//6,i%6) for i in range(n)],dtype=float)
    return a


def test_reference_preprocessing_missing_and_identity():
    a,b=raw_sample(1),raw_sample(2)
    transform=ReferenceTransform(8).fit([a],['reference'])
    original=transform.genes_
    missing_gene=original[0]
    query=b[:,b.var_names!=missing_gene].copy()
    view=transform.transform(query,'query','q')
    assert transform.genes_==original
    assert transform.fitted_names_==['reference']
    assert view.availability[0]==0
    assert np.all(view.x[:,0]==0)
    assert np.array_equal(view.barcodes,query.obs_names)
    changed=query.copy();changed.obs_names=['duplicate']*len(query)
    with pytest.raises(ValueError,match='Ambiguous'):
        transform.transform(changed,'query','q')


def test_manifest_is_resumable_and_rejects_changes(tmp_path):
    a=freeze_run(tmp_path,{'rates':(0,.1,.2),'seed':1})
    assert freeze_run(tmp_path,{'rates':(0,.1,.2),'seed':1})==a
    with pytest.raises(RuntimeError,match='changed'):
        freeze_run(tmp_path,{'rates':(0,.1,.2),'seed':2})


def test_raw_count_thinning_preserves_identity_and_recomputes_qc():
    a=raw_sample(51,n=60)
    original=a.X.copy()
    transform=ReferenceTransform(10).fit([a],['train'])
    weak=transform.transform(a,'train','g')
    strong=count_thinning_views(a,transform,'train','g',(1.,.5,.25),2,81)
    again=count_thinning_views(a,transform,'train','g',(1.,.5,.25),2,81)
    assert len(strong)==4
    assert (original!=a.X).nnz==0
    assert transform.fitted_names_==['train']
    for left,right in zip(strong,again):
        np.testing.assert_array_equal(left.x,right.x)
        np.testing.assert_array_equal(left.barcodes,weak.barcodes)
        np.testing.assert_array_equal(left.availability,weak.availability)
        raw_qc=transform.qc_scaler_.inverse_transform(left.qc)
        assert raw_qc[:,0].mean()<transform.qc_scaler_.inverse_transform(weak.qc)[:,0].mean()
        assert not np.array_equal(left.x,weak.x)
    TrainingSlice(weak,np.zeros((len(a),len(PANEL)),dtype=np.float32),strong)
    with pytest.raises(ValueError,match='Invalid raw-count'):
        count_thinning_views(a,transform,'train','g',(1.,0.),1,81)


def test_end_to_end_train_teacher_checkpoint_and_rna_only_predict(tmp_path):
    torch.set_num_threads(2)
    raw=[raw_sample(i) for i in range(3)]
    transform=ReferenceTransform(10).fit(raw[:2],['r0','r1'])
    views=[transform.transform(a,f'r{i}',f'g{i}') for i,a in enumerate(raw)]
    rng=np.random.default_rng(8)
    paired=[TrainingSlice(v,rng.normal(size=(len(v.x),len(PANEL))).astype(np.float32)) for v in views]
    for i in range(2):
        augmented=count_thinning_views(raw[i],transform,f'r{i}',f'g{i}',(1.,.5),1,2025)
        paired[i]=TrainingSlice(views[i],paired[i].target,augmented)
    router=FrozenRouter(components=4,clusters=3,per_group=16).fit([v.x for v in views[:2]],['g0','g1'])
    model_cfg=ModelConfig(n_genes=10,width=16,hidden=12,members=2,state_prototypes=3,
                          niche_prototypes=3,prototype_dim=12,experts=2,stalk=2,channels=4,
                          protein_embedding=4,dropout=0)
    train_cfg=TrainConfig(epochs=2,warmup_epochs=1,prototype_ramp_epochs=1,validate_every=1,
                          patience=3,core_size=6,min_core=3,max_core=8,threads=2,count_keep_rates=(1.,.5))
    model,report=fit(paired[:2],model_cfg,train_cfg,router,tmp_path,validation=paired[2])
    assert report['history'][-1]['teacher_updates']==2
    assert report['query_protein_used'] is False
    a,diagnostics=predict(model,views[2],router,train_cfg)
    assert a['prediction'].shape==(len(views[2].x),31)
    assert np.isfinite(a['prediction']).all()
    assert len(a['barcodes'])==len(set(a['barcodes']))
    assert diagnostics['state']['effective_codes']>0
    restored,again=fit(paired[:2],model_cfg,train_cfg,router,tmp_path,validation=paired[2])
    b,_=predict(restored,views[2],router,train_cfg)
    np.testing.assert_array_equal(a['prediction'],b['prediction'])
    assert again['selected_epoch']==report['selected_epoch']
    with pytest.raises(ValueError,match='overlaps'):
        fit(paired[:2],model_cfg,train_cfg,router,tmp_path/'bad',validation=paired[0])

