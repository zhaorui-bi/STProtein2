"""Real training tests of Teacher removal, fixed weights, equivalence and isolation."""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stprotein2.data import Cohort
from stprotein2.graph import FrozenRouter
import stprotein2.training.targets as targets
import stprotein2.research.spatial.trainer as original
from stprotein2.training.config import SpatialAnchorTrainConfig
from stprotein2.training.prediction import predict
from stprotein2.training.config import TeacherAblationTrainConfig
import stprotein2.research.teacher.trainer as trainer
import stprotein2.research.teacher.nested as nested
from test_crossfit import cohort_files, configs


def setup_case(tmp_path, target='cross_fitted'):
    cohort, _, _ = cohort_files(tmp_path)
    cfg, old = configs()
    train = TeacherAblationTrainConfig(**old.to_dict(), spatial_alpha=.5)
    train = replace(train, residual_target=target)
    names = ['s0', 's1', 's2']
    anchor = targets.get_anchor(cohort, names, 6)
    baselines = targets.build_targets(cohort, names, 6, train, anchor)
    refs = cohort.references(names, anchor[0]['transform'])
    validation = cohort.references(['s4'], anchor[0]['transform'])
    router = FrozenRouter(components=3, clusters=3).fit([s.rna.x for s in refs], names)
    return cfg, train, refs, validation, router, anchor[0]['model'], baselines


def run(case, output, mode, module=trainer):
    cfg, train, refs, validation, router, anchor, baselines = case
    return module.fit(refs, cfg, replace(train, teacher_mode=mode), router, output,
                      anchor, baselines, validation=validation)


def assert_weights_equal(left, right):
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0)


def test_mode_validation_and_student_prototypes_retained(tmp_path):
    with pytest.raises(ValueError, match='Unknown Teacher'):
        TeacherAblationTrainConfig(teacher_mode='off')
    with pytest.raises(ValueError, match='nonnegative'):
        TeacherAblationTrainConfig(state_weight=-1)
    case = setup_case(tmp_path)
    cfg, train, refs, validation, router, anchor, baselines = case
    with pytest.raises(ValueError, match='at least one'):
        trainer.fit(refs, replace(cfg, use_state=False, use_niche=False), train, router,
                    tmp_path/'invalid', anchor, baselines)
    model, report = run(case, tmp_path/'none', 'none')
    assert model.representations.state is not None and model.representations.niche is not None
    assert model.representations.state.codes.requires_grad
    assert model.representations.niche.codes.requires_grad
    assert report['teacher_parameters'] == report['total_teacher_forwards'] == 0
    # Supervised updates really reach both retained codebooks.
    cfg, _, refs, _, _, _, baseline = case
    from stprotein2.training.basic import tensor_input
    from stprotein2.graph import physical_graph
    sample = refs[0]
    graph = physical_graph(sample.rna.coordinates)
    model.train(); model.zero_grad(set_to_none=True)
    output = model(*tensor_input(sample.rna, graph, 'cpu'))
    loss, auxiliary = trainer.supervised_losses(output, torch.tensor(sample.target),
        torch.tensor(baseline[sample.rna.name].prediction), model.gamma_state, model.gamma_local)
    (loss + .2*auxiliary).backward()
    assert model.representations.state.codes.grad.abs().sum() > 0
    assert model.representations.niche.codes.grad.abs().sum() > 0


def test_ema_matches_unchanged_spatial_trainer_actual_updates(tmp_path):
    case = setup_case(tmp_path)
    cfg, train, refs, validation, router, anchor, baselines = case
    old_cfg = SpatialAnchorTrainConfig(**{k:v for k,v in train.to_dict().items() if k != 'teacher_mode'})
    expected, before = original.fit(refs, cfg, old_cfg, router, tmp_path/'old', anchor, baselines,
                                     validation=validation)
    actual, after = run(case, tmp_path/'ema', 'ema')
    assert_weights_equal(expected, actual)
    assert before['selected_epoch'] == after['selected_epoch']
    assert before['reference_validation_mse'] == after['reference_validation_mse']
    for a,b in zip(before['history'], after['history']):
        for key in ['loss','protein','state_proto','niche_proto','last_gradient_norm','teacher_updates']:
            assert a[key] == b[key], key
    assert after['total_teacher_forwards'] == after['total_teacher_center_updates'] == 6


@pytest.mark.parametrize('mode', ['none','no_loss','fixed','ema'])
def test_each_mode_actual_post_initialization_resume_and_teacher_events(tmp_path, monkeypatch, mode):
    case = setup_case(tmp_path)
    if mode == 'none':
        monkeypatch.setattr(trainer, 'EMATeacher', lambda *a,**k: pytest.fail('No-Teacher mode constructed a Teacher'))
    expected, report = run(case, tmp_path/'full', mode)
    original_save = trainer.save_torch
    def stop_after_teacher_initialized(path, value):
        original_save(path, value)
        if path.name == 'resume.pt' and value['epoch'] == 2:
            raise RuntimeError('intentional post-initialization interruption')
    monkeypatch.setattr(trainer, 'save_torch', stop_after_teacher_initialized)
    with pytest.raises(RuntimeError, match='intentional'):
        run(case, tmp_path/'resumed', mode)
    monkeypatch.setattr(trainer, 'save_torch', original_save)
    actual, other = run(case, tmp_path/'resumed', mode)
    assert_weights_equal(expected, actual)
    assert report['reference_validation_mse'] == other['reference_validation_mse']
    assert report['anchor_unchanged'] and other['anchor_unchanged']
    a = torch.load(tmp_path/'full/resume.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed/resume.pt', weights_only=False)
    if mode == 'none':
        assert a['teacher'] is b['teacher'] is None
        assert other['total_teacher_forwards'] == 0
    else:
        for key, value in a['teacher'].items():
            torch.testing.assert_close(value, b['teacher'][key], rtol=0, atol=0)
        assert other['total_teacher_forwards'] == other['total_teacher_center_updates'] == 6
        assert a['teacher']['state_center'].abs().sum() > 0
        assert a['teacher']['niche_center'].abs().sum() > 0
        assert int(a['teacher']['update_count']) == (0 if mode == 'fixed' else 6)
    if mode == 'fixed':
        assert report['initial_teacher_parameter_signature'] == report['final_training_teacher_parameter_signature']
    if mode in {'ema','no_loss'}:
        assert report['initial_teacher_parameter_signature'] != report['final_training_teacher_parameter_signature']
    assert other['teacher_distillation_applied'] == (mode in {'ema','fixed'})
    loaded = nested.load_model(tmp_path/'resumed/model.pt', 'cpu')
    assert_weights_equal(actual, loaded)
    with pytest.raises(RuntimeError, match='manifest|configuration|differs|changed'):
        run(case, tmp_path/'resumed', 'ema' if mode != 'ema' else 'fixed')


def test_no_loss_and_no_teacher_have_exact_student_weights_and_predictions(tmp_path):
    case = setup_case(tmp_path, 'in_fit')
    plain, a = run(case, tmp_path/'none', 'none')
    diagnostic, b = run(case, tmp_path/'no_loss', 'no_loss')
    assert_weights_equal(plain, diagnostic)
    assert a['Student_parameters'] == b['Student_parameters']
    assert a['teacher_parameters'] == 0 < b['teacher_parameters']
    for left,right in zip(a['history'],b['history']):
        assert left['loss'] == right['loss']
        assert left['protein'] == right['protein']
    cfg, train, _, validation, router, _, _ = case
    x,_ = predict(plain,validation[0].rna,router,train)
    y,_ = predict(diagnostic,validation[0].rna,router,train)
    for name in x: np.testing.assert_array_equal(x[name],y[name])


def test_nested_31_fits_query_ADT_absent_recovery_and_separate_scoring(tmp_path, monkeypatch):
    cohort, registry, groups = cohort_files(tmp_path)
    cfg, old = configs()
    train = TeacherAblationTrainConfig(**old.to_dict(), spatial_alpha=.5, teacher_mode='none')
    train = replace(train,epochs=2)
    mapping=tmp_path/'groups.json';mapping.write_text(json.dumps({'groups':groups}))
    spec=tmp_path/'config.json';spec.write_text(json.dumps({'model':cfg.to_dict(),'train':train.to_dict()}))
    query_path=Path(cohort.specs['s6']['adt']);query_bytes=query_path.read_bytes();query_path.unlink()
    output=tmp_path/'outer'
    args=argparse.Namespace(registry=str(registry),groups=str(mapping),config=str(spec),query='s6',
        output=str(output),anchor_cache=str(tmp_path/'anchors'),device='cpu',reuse_anchor_cache=[])
    original_reader=Cohort.measured_protein
    def guarded(self,name):
        assert name!='s6';return original_reader(self,name)
    monkeypatch.setattr(Cohort,'measured_protein',guarded)
    nested.run_nested(args)
    reports=list(output.rglob('training.json'));assert len(reports)==31
    for path in reports:
        row=json.loads(path.read_text())
        assert 's6' not in row['reference_names']+row['inner_validation']
        assert row['teacher_parameters']==row['total_teacher_forwards']==0
    for path in output.glob('gate_oof_*/component_losses.json'):
        held=json.loads(path.read_text())['held_group']
        for report in path.parent.rglob('training.json'):
            row=json.loads(report.read_text())
            assert held not in row['reference_groups']+row['inner_validation_groups']
    manifest=json.loads((output/'manifest.json').read_text())['protocol']
    assert str(query_path) not in manifest['inputs']
    prediction=output/'predictions/s6/prediction.npz';before=prediction.read_bytes()
    nested.run_nested(args)
    (output/'prediction_complete.json').unlink();nested.run_nested(args)
    assert prediction.read_bytes()==before
    query_path.write_bytes(query_bytes)
    monkeypatch.setattr(Cohort,'measured_protein',original_reader)
    path=Path(__file__).resolve().parents[1]/'scripts/score_extended_baseline.py'
    module_spec=importlib.util.spec_from_file_location('teacher_separate_score',path)
    scorer=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(scorer)
    scores=scorer.score(registry,output)
    assert set(scores)=={'s6'} and (output/'score_complete.json').exists()
    artifact=next(output.glob('gate_oof_*/inner_*/model.pt'))
    with artifact.open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError,match='artifact identity'):
        nested.run_nested(args)
