"""OOF provenance, true-Y auxiliary gradients, deployment validation and recovery."""
import argparse
from dataclasses import replace
import json

import anndata as ad
import joblib
import numpy as np
import pytest
from scipy import sparse
import torch
from torch.nn import functional as F

from stprotein2.config import ModelConfig
from stprotein2.data import Cohort, PANEL
from stprotein2.graph import FrozenRouter, physical_graph
from stprotein2.research.joint.cli import score_run
from stprotein2.training.basic import predict, tensor_input
from stprotein2.models.anchor import AnchoredSTProtein2
import stprotein2.research.anchor.trainer as frozen_trainer
from stprotein2.training.config import ResidualTrainConfig
import stprotein2.research.residuals.nested as nested
import stprotein2.training.residual_targets as targets
import stprotein2.research.residuals.trainer as trainer


def cohort_files(path):
    rng = np.random.default_rng(801)
    entries, groups = [], {}
    for i, group in enumerate(['a', 'b', 'c', 'c', 'd', 'e', 'query']):
        name = f's{i}'
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3+i/3, (12, 8)).astype(np.float32)))
        rna.obs_names = [f'{name}_b{j}' for j in range(12)]
        rna.var_names = [f'g{j}' for j in range(8)]
        rna.obsm['spatial'] = np.array([(j//4,j%4) for j in range(12)], dtype=float)
        adt = ad.AnnData(sparse.csr_matrix(rng.poisson(2+i, (12, 31)).astype(np.float32)))
        adt.obs_names = rna.obs_names; adt.var_names = PANEL
        rp, ap = path/f'{name}_rna.h5ad', path/f'{name}_adt.h5ad'
        rna.write_h5ad(rp); adt.write_h5ad(ap)
        entries.append({'name':name,'rna':str(rp),'adt':str(ap)}); groups[name]=group
    registry = path/'registry.json'; registry.write_text(json.dumps(entries))
    cohort = Cohort(registry, groups)
    cohort.anchor_cache = path/'anchors'
    return cohort, registry, groups


def configs():
    model = ModelConfig(n_genes=6, width=12, hidden=8, members=2, dropout=.1,
                        state_prototypes=3, niche_prototypes=3, prototype_dim=8,
                        local_operator='csnn', experts=1, stalk=2, channels=2, protein_embedding=2)
    train = ResidualTrainConfig(epochs=3, warmup_epochs=1, prototype_ramp_epochs=1,
                        validate_every=1, threads=2, execution_mode='full')
    return model, train


def test_crossfit_feasibility_and_matched_augmentation_rejected_before_data(tmp_path):
    _, cfg = configs()
    assert cfg.first_checkpoint_epoch == 2
    with pytest.raises(ValueError, match='unaugmented'):
        replace(cfg, mask_rates=(0., .1))
    with pytest.raises(ValueError, match='unaugmented'):
        replace(cfg, count_keep_rates=(1., .5))
    with pytest.raises(ValueError, match='after-ramp'):
        replace(cfg, checkpoint_stage='any')
    groups = tmp_path/'groups.json'; groups.write_text(json.dumps({'groups':{'a':'a','b':'b','c':'c','q':'q'}}))
    spec = tmp_path/'config.json'; spec.write_text('{}')
    args = argparse.Namespace(config=str(spec),groups=str(groups),registry='does_not_exist')
    with pytest.raises(ValueError, match='at least six'):
        nested.run_nested(args)


def test_oof_anchor_excludes_target_group_from_features_coefficients_and_cv(tmp_path, monkeypatch):
    cohort, _, groups = cohort_files(tmp_path)
    names = ['s0','s1','s2','s3']
    full = targets.get_anchor(cohort, names, 6)
    real_get = targets.get_anchor
    calls = []

    def guarded_get(c, allowed, *args):
        real_rna, real_protein = c.rna, c.measured_protein
        def only_allowed_rna(name):
            assert name in allowed
            return real_rna(name)
        def only_allowed_protein(name):
            assert name in allowed
            return real_protein(name)
        c.rna, c.measured_protein = only_allowed_rna, only_allowed_protein
        try:
            bundle, path = real_get(c, allowed, *args)
            calls.append(list(allowed))
            return bundle, path
        finally:
            c.rna, c.measured_protein = real_rna, real_protein
    monkeypatch.setattr(targets, 'get_anchor', guarded_get)
    oof = targets.build_targets(cohort, names, 6, 'cross_fitted', full)
    assert len(calls) == 3
    assert oof['s2'].provenance['anchor_reference_names'] == ['s0','s1']
    assert oof['s3'].provenance['anchor_reference_names'] == ['s0','s1']
    for name, baseline in oof.items():
        provenance = baseline.provenance
        assert groups[name] not in provenance['anchor_reference_groups']
        for fold in provenance['anchor_cv']:
            assert all(groups[n] != groups[name] for n in fold['training']+fold['validation'])
            assert not set(fold['training']) & set(fold['validation'])
    assert not set(cohort.protein_reads) & {'s4','s5','s6'}
    references = cohort.references(names, full[0]['transform'])
    targets.validate_targets(references, oof, 'cross_fitted')
    control = targets.build_targets(cohort, names, 6, 'in_fit', full)
    for sample in references:
        expected = targets.predict_anchor(full[0], sample.rna)
        np.testing.assert_array_equal(control[sample.rna.name].prediction, expected)
    assert any(not np.array_equal(oof[n].prediction, control[n].prediction) for n in names)
    corrupted = dict(oof)
    corrupted['s2'] = replace(oof['s2'], barcodes=tuple(reversed(oof['s2'].barcodes)))
    with pytest.raises(ValueError, match='row identity'):
        targets.validate_targets(references, corrupted, 'cross_fitted')


def test_matched_loss_gradient_parity_and_true_protein_auxiliary(tmp_path):
    cohort, _, _ = cohort_files(tmp_path)
    names = ['s0','s1','s2']
    bundle, _ = targets.get_anchor(cohort, names, 6)
    sample = cohort.references(['s0'],bundle['transform'])[0]
    cfg, _ = configs(); cfg = replace(cfg, dropout=0)
    torch.manual_seed(21)
    model = AnchoredSTProtein2(cfg)
    model.anchor.set_coefficients(bundle['model'].coef_,bundle['model'].intercept_)
    # Nonzero corrections make this more than a fallback-only check.
    with torch.no_grad():
        model.state_readout[-1].weight.normal_(0,.1)
        model.niche_readout[-1].weight.normal_(0,.1)
        model.expert_readout.weight.normal_(0,.1)
    args = tensor_input(sample.rna,physical_graph(sample.rna.coordinates),'cpu')
    y = torch.tensor(sample.target)
    out = model(*args)
    old_main = F.mse_loss(out['prediction'],y)
    old_aux = F.mse_loss(out['auxiliary_members'],y[:,None,:].expand_as(out['auxiliary_members']))
    (old_main + .2*old_aux).backward()
    expected = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    expected_direct = expected['direct.weight'].clone()
    model.zero_grad(set_to_none=True)
    out = model(*args)
    main, aux = trainer.supervised_losses(out,y,out['base'].detach(),model.gamma_state,model.gamma_local)
    torch.testing.assert_close(main,old_main,rtol=0,atol=0)
    torch.testing.assert_close(aux,old_aux,rtol=0,atol=0)
    (main+.2*aux).backward()
    for name,p in model.named_parameters():
        if name in expected:torch.testing.assert_close(p.grad,expected[name],rtol=0,atol=0)
    model.zero_grad(set_to_none=True)
    out = model(*args)
    cf_main, cf_aux = trainer.supervised_losses(out,y,out['base'].detach()+1.7,model.gamma_state,model.gamma_local)
    (cf_main+.2*cf_aux).backward()
    torch.testing.assert_close(model.direct.weight.grad,expected_direct,rtol=0,atol=0)
    assert not torch.equal(model.state_readout[-1].weight.grad,expected['state_readout.2.weight'])
    assert all(p.grad is None for p in model.anchor.parameters())


def test_crossfit_interruption_resume_and_changed_targets_rejected(tmp_path,monkeypatch):
    cohort, _, _ = cohort_files(tmp_path)
    cfg, train_cfg = configs()
    names = ['s0','s1','s2']
    full = targets.get_anchor(cohort,names,6)
    baselines = targets.build_targets(cohort,names,6,'cross_fitted',full)
    references = cohort.references(names,full[0]['transform'])
    validation = cohort.references(['s4'],full[0]['transform'])
    router = FrozenRouter(components=3,clusters=3).fit([s.rna.x for s in references],[s.rna.group for s in references])
    model, report = trainer.fit(references,cfg,train_cfg,router,tmp_path/'continuous',full[0]['model'],baselines,validation=validation)
    real_save = trainer.save_torch
    def interrupt(path,value):
        real_save(path,value)
        if path.name=='resume.pt' and value['epoch']==1:raise RuntimeError('injected checkpoint interruption')
    monkeypatch.setattr(trainer,'save_torch',interrupt)
    with pytest.raises(RuntimeError,match='injected checkpoint'):
        trainer.fit(references,cfg,train_cfg,router,tmp_path/'resumed',full[0]['model'],baselines,validation=validation)
    monkeypatch.setattr(trainer,'save_torch',real_save)
    resumed, again = trainer.fit(references,cfg,train_cfg,router,tmp_path/'resumed',full[0]['model'],baselines,validation=validation)
    assert report['selected_epoch']==again['selected_epoch']>=2
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
    arrays,_=predict(resumed,validation[0].rna,router,train_cfg,'cpu')
    mse=float(np.mean((arrays['prediction']-validation[0].target)**2,dtype=np.float64))
    assert mse==again['reference_validation_mse']
    assert again['validation_baseline']=='full_allowed_training_anchor'
    frozen = {k:v.clone() for k,v in resumed.anchor.state_dict().items()}
    resumed.set_gates(np.zeros(31),np.zeros(31))
    closed,_=predict(resumed,validation[0].rna,router,train_cfg,'cpu')
    np.testing.assert_array_equal(closed['prediction'],closed['base'])
    assert all(torch.equal(frozen[k],v) for k,v in resumed.anchor.state_dict().items())
    altered = dict(baselines)
    altered['s0'] = replace(baselines['s0'],prediction=baselines['s0'].prediction+.1)
    with pytest.raises(RuntimeError,match='changed'):
        trainer.fit(references,cfg,train_cfg,router,tmp_path/'resumed',full[0]['model'],altered,validation=validation)
    references[0].target[0,0]+=.1
    with pytest.raises(RuntimeError,match='changed'):
        trainer.fit(references,cfg,train_cfg,router,tmp_path/'resumed',full[0]['model'],baselines,validation=validation)


def test_C0_matches_frozen_anchor_schedule_with_identical_unaugmented_data(tmp_path):
    cohort, _, _ = cohort_files(tmp_path)
    cfg, train_cfg = configs()
    train_cfg = replace(train_cfg, epochs=2, residual_target='in_fit')
    names = ['s0','s1','s2']
    full = targets.get_anchor(cohort,names,6)
    baselines = targets.build_targets(cohort,names,6,'in_fit',full)
    refs = cohort.references(names,full[0]['transform'])
    router = FrozenRouter(components=3,clusters=3).fit([s.rna.x for s in refs],[s.rna.group for s in refs])
    expected, before = frozen_trainer.fit(refs,cfg,train_cfg,router,tmp_path/'old_control',full[0]['model'])
    actual, after = trainer.fit(refs,cfg,train_cfg,router,tmp_path/'C0',full[0]['model'],baselines)
    for name,value in expected.state_dict().items():
        torch.testing.assert_close(value,actual.state_dict()[name],rtol=0,atol=0)
    assert [r['loss'] for r in before['history']]==[r['loss'] for r in after['history']]
    assert [r['teacher_updates'] for r in before['history']]==[r['teacher_updates'] for r in after['history']]


def test_six_group_nested_cf_targets_exclude_gate_and_epoch_groups_and_score_separately(tmp_path,monkeypatch):
    _,registry,groups=cohort_files(tmp_path)
    cfg,train_cfg=configs();train_cfg=replace(train_cfg,epochs=2)
    mapping=tmp_path/'groups.json';mapping.write_text(json.dumps({'groups':groups}))
    spec=tmp_path/'config.json';spec.write_text(json.dumps({'model':cfg.to_dict(),'train':train_cfg.to_dict()}))
    output=tmp_path/'run'
    args=argparse.Namespace(registry=str(registry),groups=str(mapping),config=str(spec),query='s6',
                output=str(output),anchor_cache=str(tmp_path/'anchors'),device='cpu',reuse_anchor_cache=[])
    real_reader=Cohort.measured_protein
    def guard(self,name):
        assert name!='s6'
        return real_reader(self,name)
    monkeypatch.setattr(Cohort,'measured_protein',guard)
    nested.run_nested(args,prepare_only=True)
    prepared=json.loads((output/'anchors_prepared.json').read_text())
    assert len(prepared['allowed_reference_subsets'])==26
    assert not prepared['query_protein_read']
    nested.run_nested(args)
    assert not (output/'development_scores.json').exists()
    reports=list(output.rglob('training.json'))
    assert len(reports)==31
    for report_path in reports:
        row=json.loads(report_path.read_text())
        assert row['variant']=='residual_CF0' and row['selected_corrections_trained']
        assert row['selected_teacher_updates']>0 and row['anchor_unchanged']
        assert 's6' not in row['reference_names']+row['inner_validation']
        baselines=json.loads((report_path.parent/'residual_baselines.json').read_text())
        for name,baseline in baselines.items():
            lineage=baseline['provenance']
            assert set(lineage['anchor_reference_names']) <= set(row['reference_names'])
            assert groups[name] not in lineage['anchor_reference_groups']
            assert not set(lineage['anchor_reference_names']) & set(row['inner_validation'])
    for evidence_path in output.glob('gate_oof_*/component_losses.json'):
        held=json.loads(evidence_path.read_text())['held_group']
        for target_path in evidence_path.parent.rglob('residual_baselines.json'):
            for baseline in json.loads(target_path.read_text()).values():
                assert held not in baseline['provenance']['anchor_reference_groups']
    completion=(output/'prediction_complete.json').read_bytes()
    nested.run_nested(args)
    assert completion==(output/'prediction_complete.json').read_bytes()
    # Simulate interruption after all fits but before the outer completion
    # marker: reuse exact gate evidence and completed fits without rewriting
    # their provenance or changing the actual RNA-only predictions.
    previous=np.load(output/'predictions/s6/prediction.npz')['prediction'].copy()
    (output/'prediction_complete.json').unlink()
    nested.run_nested(args)
    np.testing.assert_array_equal(previous,np.load(output/'predictions/s6/prediction.npz')['prediction'])
    monkeypatch.setattr(Cohort,'measured_protein',real_reader)
    score_run(args)
    assert (output/'development_scores.json').exists()
    # A completed run must still detect a changed deployment artifact.
    with (output/'model.pt').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(RuntimeError,match='deployment artifact changed'):
        nested.run_nested(args)
    (output/'prediction_complete.json').unlink()
    gate_path=next(output.glob('gate_oof_*/component_losses.json'))
    record=json.loads(gate_path.read_text());record['group_mean_losses'][0][0]+=1
    gate_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='gate evidence changed'):
        nested.run_nested(args)
