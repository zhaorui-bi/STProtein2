"""Actual training, interrupted recovery, saved inference and nested isolation."""
from dataclasses import replace
import json

import joblib
import numpy as np
import pytest
import torch

from stprotein2.execution.memory import CheckpointRepresentations, WeightModulatedEnsembleLinear
from stprotein2.training.prediction import predict
import research.teacher_recovery.trainer as previous
import stprotein2.training.trainer as trainer
import research.strict_nested as nested
from stprotein2.training.config import BalancedTrainConfig
import test_teacher_ablation as original


def balanced_case(tmp_path, **options):
    cfg, old, refs, validation, router, anchor, baselines = original.setup_case(tmp_path)
    train = BalancedTrainConfig(**old.to_dict(), router_components=3, router_clusters=3, **options)
    return cfg, train, refs, validation, router, anchor, baselines


def test_policy_validation_and_pooled_original_math(tmp_path):
    for kwargs in ({"initialization":"unknown"}, {"router_clusters":True},
                   {"weight_modulated_input":1}, {"initialization":"pooled","prototype_per_source":15}):
        with pytest.raises(ValueError):
            BalancedTrainConfig(**kwargs)
    case = original.setup_case(tmp_path)
    old, before = original.run(case, tmp_path/'previous', 'ema', module=previous)
    cfg, previous_cfg, *rest = case
    train = BalancedTrainConfig(**previous_cfg.to_dict(), initialization='pooled',
        weight_modulated_input=False, checkpoint_representations=False)
    new, after = original.run((cfg, train, *rest), tmp_path/'new', 'ema', module=trainer)
    original.assert_weights_equal(old, new)
    assert before['reference_validation_mse'] == after['reference_validation_mse']


@pytest.mark.parametrize('initialization,mode,interrupted_epoch', [
    ('pooled','ema',2), ('hierarchical','ema',2), ('hierarchical','none',2), ('hierarchical','none',1)])
def test_registered_initialization_policy_survives_training_and_restore(tmp_path, monkeypatch, initialization, mode, interrupted_epoch):
    case = balanced_case(tmp_path, initialization=initialization)
    expected, report = original.run(case, tmp_path/'full', mode, module=trainer)
    assert report['execution']['policy'] == {'weight_modulated_input':True,'checkpoint_representations':True}
    assert isinstance(expected.representations, CheckpointRepresentations)
    assert any(isinstance(module, WeightModulatedEnsembleLinear) for module in expected.modules())
    save = trainer.save_torch
    def stop(path, value):
        save(path, value)
        if path.name == 'resume.pt' and value['epoch'] == interrupted_epoch:
            raise RuntimeError('registered interruption')
    monkeypatch.setattr(trainer,'save_torch',stop)
    with pytest.raises(RuntimeError,match='registered interruption'):
        original.run(case,tmp_path/'resumed',mode,module=trainer)
    monkeypatch.setattr(trainer,'save_torch',save)
    actual, restored = original.run(case,tmp_path/'resumed',mode,module=trainer)
    original.assert_weights_equal(expected,actual)
    assert report['initialization'] == restored['initialization']
    assert report['reference_validation_mse'] == restored['reference_validation_mse']
    left = torch.load(tmp_path/'full/resume.pt',weights_only=False)
    right = torch.load(tmp_path/'resumed/resume.pt',weights_only=False)
    if mode == 'ema':
        for name,value in left['teacher'].items():
            torch.testing.assert_close(value,right['teacher'][name],rtol=0,atol=0)
    else:
        assert left['teacher'] is right['teacher'] is None
    if initialization == 'hierarchical':
        record = restored['initialization']['sampling']
        assert record['reference_names'] == ['s0','s1','s2']
        assert record['reference_groups'] == ['a','b','c']
        assert restored['initialization']['stages'][1]['niche_uses_initialized_state']
    loaded = nested.load_model(tmp_path/'resumed/model.pt','cpu')
    assert isinstance(loaded.representations,CheckpointRepresentations)
    original.assert_weights_equal(actual,loaded)
    cfg, train, _, validation, router, _, _ = case
    a,_=predict(actual,validation[0].rna,router,train)
    b,_=predict(loaded,validation[0].rna,router,train)
    assert np.any(a['state_delta'] != 0) and np.any(a['local_delta'] != 0)
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])
    monkeypatch.setattr(trainer,'initialize_balanced_prototypes',lambda *a,**k: pytest.fail('Repeated initialization'))
    reused, _ = original.run(case,tmp_path/'resumed',mode,module=trainer)
    original.assert_weights_equal(actual,reused)
    changed = (cfg,replace(train,checkpoint_representations=False),*case[2:])
    with pytest.raises(RuntimeError,match='manifest|configuration|differs|changed'):
        original.run(changed,tmp_path/'resumed',mode,module=trainer)


def test_full_31_fit_nested_initializers_exclude_query_and_gate_held_source(tmp_path, monkeypatch):
    def make_config(**kwargs):
        return BalancedTrainConfig(**kwargs,router_components=3,router_clusters=3)
    monkeypatch.setattr(original,'TeacherAblationTrainConfig',make_config)
    monkeypatch.setattr(original,'nested',nested)
    original.test_nested_31_fits_query_ADT_absent_recovery_and_separate_scoring(tmp_path,monkeypatch)
    reports=list((tmp_path/'outer').rglob('training.json'))
    assert len(reports)==31
    for path in reports:
        report=json.loads(path.read_text())
        prep=joblib.load(path.with_name('preprocessing.joblib'))
        expected=sorted(report['reference_names'])
        assert prep['router_sampling']['reference_names']==expected
        assert report['initialization']['sampling']['reference_names']==expected
        assert not set(report['inner_validation']) & set(expected)
        assert 's6' not in expected
        if path.parent.parent.name.startswith('gate_oof_'):
            gate=json.loads((path.parent.parent/'component_losses.json').read_text())
            assert gate['held_group'] not in prep['router_sampling']['reference_groups']
            assert gate['held_group'] not in report['initialization']['sampling']['reference_groups']
