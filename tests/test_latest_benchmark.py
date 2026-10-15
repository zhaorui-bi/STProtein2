"""Exercise README feasibility without weakening gate or label exclusion rules."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from stprotein2.data import Cohort
from stprotein2.training.config import BalancedTrainConfig
import research.cohort_v1.nested as nested
from test_crossfit import cohort_files, configs


def test_four_source_actual_training_closed_gates_and_missing_query_ADT(tmp_path,monkeypatch):
    cohort,registry,allgroups=cohort_files(tmp_path)
    names=['s0','s1','s2','s3','s4']
    groups={n:allgroups[n] for n in names}
    records=json.loads(registry.read_text())
    registry=tmp_path/'primary_registry.json'
    registry.write_text(json.dumps([r for r in records if r['name'] in names]))
    mapping=tmp_path/'groups.json';mapping.write_text(json.dumps({'groups':groups}))
    cfg,old=configs()
    train=BalancedTrainConfig(**replace(old,residual_target='in_fit').to_dict(),teacher_mode='none',
        router_components=3,router_clusters=3,spatial_alpha=.5)
    spec=tmp_path/'config.json';spec.write_text(json.dumps({'model':cfg.to_dict(),'train':train.to_dict()}))
    query=Path(cohort.specs['s4']['adt']);raw=query.read_bytes();query.unlink()
    reader=Cohort.measured_protein
    def guarded(self,name):
        assert name!='s4'
        return reader(self,name)
    monkeypatch.setattr(Cohort,'measured_protein',guarded)
    args=argparse.Namespace(config=str(spec),groups=str(mapping),registry=str(registry),query='s4',
        output=str(tmp_path/'outer'),anchor_cache=str(tmp_path/'anchors'),device='cpu',reuse_anchor_cache=[],
        cuda_allocator_gib=4.)
    nested.run_nested(args)
    outer=Path(args.output)
    reports=list(outer.rglob('training.json'))
    assert len(reports)==4
    assert not list(outer.glob('gate_oof_*'))
    for p in reports:
        row=json.loads(p.read_text());prep=joblib.load(p.with_name('preprocessing.joblib'))
        assert 's4' not in row['reference_names']+row['inner_validation']
        assert 'd' not in row['reference_groups']+row['inner_validation_groups']
        assert prep['router_sampling']['reference_names']==sorted(row['reference_names'])
        assert row['initialization']['sampling']['reference_names']==sorted(row['reference_names'])
    prediction=outer/'predictions/s4/prediction.npz'
    with np.load(prediction,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['prediction'],archive['base'])
        assert not archive['gamma_state'].any() and not archive['gamma_local'].any()
        assert np.any(archive['state_delta']!=0) and np.any(archive['local_delta']!=0)
    before=prediction.read_bytes();nested.run_nested(args);assert prediction.read_bytes()==before
    query.write_bytes(raw)
    monkeypatch.setattr(Cohort,'measured_protein',reader)
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/score_extended_baseline.py'
    spec=importlib.util.spec_from_file_location('latest_benchmark_scorer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert set(module.score(registry,outer))=={'s4'}
    extended = outer / 'extended_report'
    scores = module.score(registry, outer, extended=True, report_dir=extended)
    assert 'moran_alignment_MAE' in scores['s4'] and 'ari' in scores['s4']
    assert module.score(registry, outer, extended=True, report_dir=extended) == scores
    (extended / 'development_scores.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Artifact missing or changed'):
        module.score(registry, outer, extended=True, report_dir=extended)


def test_CF0_still_requires_six_groups_before_accessing_registry(tmp_path):
    mapping=tmp_path/'groups.json'
    mapping.write_text(json.dumps({'groups':{str(i):str(i) for i in range(4)}}))
    spec=tmp_path/'config.json';spec.write_text(json.dumps({'train':{'residual_target':'cross_fitted'}}))
    args=argparse.Namespace(config=str(spec),groups=str(mapping),registry='must_not_open')
    with pytest.raises(ValueError,match='Insufficient cohort'):
        nested.run_nested(args)
