"""Actual nested fitting, interrupted training and standalone neural use."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip('torch')

from stprotein2 import training
from stprotein2.anchor import fit_anchor
from stprotein2.data import Cohort, PANEL
from stprotein2.deployment import export_clean
from stprotein2.io import sha256
from stprotein2.neural import ResidualConfig
from stprotein2.pipeline import load_residual, predict_file
from stprotein2.selection import train
from stprotein2.training import TrainConfig
from evaluation.__main__ import score
from test_clean_core import cohort


def test_interrupted_fit_resumes_exactly_and_completed_fit_rejects_damage(tmp_path, monkeypatch):
    registry, groups = cohort(tmp_path)
    source = Cohort(registry, json.loads(groups.read_text())['groups'])
    anchor, _ = fit_anchor(source, ['a', 'b'], 8, tmp_path / 'anchors')
    refs = source.references(['a', 'b'], anchor['transform'])
    held = source.references(['c'], anchor['transform'])
    cfg = ResidualConfig(n_genes=8, width=8, hidden=6, members=2, residual_width=10, residual_head='neighbor_mlp')
    budget = TrainConfig(epochs=4, warmup_epochs=1, checkpoint_delay=1, validate_every=1, seed=73)
    expected, report = training.fit(refs, held, anchor, cfg, budget, tmp_path / 'uninterrupted')
    original_save = training.save_torch
    def interrupt_after_resume(path, value):
        original_save(path, value)
        if Path(path).name == 'resume.pt' and value['epoch'] == 1:
            raise InterruptedError('simulated interruption after an actual saved update')
    monkeypatch.setattr(training, 'save_torch', interrupt_after_resume)
    with pytest.raises(InterruptedError):
        training.fit(refs, held, anchor, cfg, budget, tmp_path / 'resumed')
    monkeypatch.setattr(training, 'save_torch', original_save)
    actual, resumed_report = training.fit(refs, held, anchor, cfg, budget, tmp_path / 'resumed')
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], atol=0, rtol=0)
    assert resumed_report['selected_epoch'] == report['selected_epoch']
    for a, b in zip(resumed_report['history'], report['history'], strict=True):
        for key in ('loss', 'protein', 'auxiliary', 'last_gradient_norm', 'sampled_slices', 'reference_validation_mse'):
            assert a[key] == b[key]
    checksum = sha256(tmp_path / 'resumed/resume.pt')
    training.fit(refs, held, anchor, cfg, budget, tmp_path / 'resumed')
    assert sha256(tmp_path / 'resumed/resume.pt') == checksum
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high' if precision != 'high' else 'highest')
        with pytest.raises(RuntimeError, match='code/configuration/inputs changed'):
            training.fit(refs, held, anchor, cfg, budget, tmp_path / 'resumed')
    finally:
        torch.set_float32_matmul_precision(precision)
    with (tmp_path / 'resumed/training.json').open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='artifact identity changed'):
        training.fit(refs, held, anchor, cfg, budget, tmp_path / 'resumed')


def six_sources(folder):
    folder.mkdir()
    rng = np.random.default_rng(615)
    entries, groups = [], {}
    query = None
    for i, (name, group) in enumerate([('a1', 'a'), ('a2', 'a'), ('b', 'b'), ('c', 'c'),
                                      ('d', 'd'), ('e', 'e'), ('query', 'q'), ('same_query_source', 'q')]):
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (9 + i % 3, 10)).astype(np.float32)))
        rna.obs_names = [f'{name}_{j}' for j in range(len(rna))]
        rna.var_names = [f'g{j}' for j in range(10)]
        rna.obsm['spatial'] = np.array([(j // 4, j % 4) for j in range(len(rna))], dtype=float)
        rp, ap = folder / f'{name}_rna.h5ad', folder / f'{name}_adt.h5ad'
        # Neither held sample's expression file exists during training.
        if group != 'q':
            rna.write_h5ad(rp)
            protein = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (len(rna), 31)).astype(np.float32)))
            protein.obs_names, protein.var_names = rna.obs_names, PANEL
            protein.write_h5ad(ap)
        elif name == 'query':
            query = rna
        entries.append({'name': name, 'rna': str(rp), 'adt': str(ap)})
        groups[name] = group
    registry, group_file = folder / 'registry.json', folder / 'groups.json'
    registry.write_text(json.dumps(entries)); group_file.write_text(json.dumps({'groups': groups}))
    return registry, group_file, query


def test_cli_31_nested_fits_export_rna_prediction_and_independent_score(tmp_path, monkeypatch):
    registry, groups_path, query = six_sources(tmp_path / 'references')
    cfg = ResidualConfig(n_genes=8, width=8, hidden=6, members=2, residual_width=10, residual_head='neighbor_mlp')
    budget = TrainConfig(epochs=2, warmup_epochs=0, checkpoint_delay=1, validate_every=1, patience=2, seed=42)
    config_file = tmp_path / 'tiny_test_budget.json'
    config_file.write_text(json.dumps({'model': {'width': 8, 'hidden': 6, 'members': 2, 'residual_width': 10},
                                      'train': {'epochs': 2, 'warmup_epochs': 0, 'checkpoint_delay': 1,
                                                'validate_every': 1, 'patience': 2, 'seed': 42}}))
    run = tmp_path / 'nested'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    with (tmp_path / 'training.log').open('w') as log:
        completed = subprocess.run([sys.executable, '-m', 'stprotein2', 'train', '--model', 'neighbor_mlp',
                                    '--config', str(config_file), '--registry', str(registry), '--groups', str(groups_path),
                                    '--query', 'query', '--genes', '8', '--output', str(run)],
                                   cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, (tmp_path / 'training.log').read_text()
    audit = json.loads((run / 'training_complete.json').read_text())
    assert audit['fit_count'] == 31
    assert set(audit['protein_reads']) == {'a1', 'a2', 'b', 'c', 'd', 'e'}
    for relative in audit['fit_markers']:
        protocol = json.loads((run / relative).with_name('manifest.json').read_text())['protocol']
        used = protocol['reference_identity'] + protocol['validation_identity']
        assert all(s['group'] != 'q' for s in used)
        if relative.startswith('gate_oof_'):
            gate_group = ['a', 'b', 'c', 'd', 'e'][int(relative.split('/')[0].removeprefix('gate_oof_'))]
            assert all(s['group'] != gate_group for s in used)
    model = load_residual(run)
    assert model.readout[-1].weight.abs().max() > 0
    assert all(not n.startswith('direct.') for n in model.state_dict())
    def no_more_labels(self, name):
        raise RuntimeError('Completed training or deployment must not reopen protein matrices')
    original_reader = Cohort.measured_protein
    monkeypatch.setattr(Cohort, 'measured_protein', no_more_labels)
    train(registry, groups_path, run, query='query', model_config=cfg, config=budget)
    bundle = tmp_path / 'portable'
    export_clean(run, bundle)
    assert len(list(bundle.iterdir())) == 6
    shutil.rmtree(run); shutil.rmtree(tmp_path / 'references')
    config_file.unlink()
    query_path = tmp_path / 'query.h5ad'; query.write_h5ad(query_path)
    prediction = tmp_path / 'prediction'
    predict_file(bundle, query_path, prediction, sample='query')
    with np.load(prediction / 'prediction.npz', allow_pickle=False) as result:
        assert result['prediction'].shape == (len(query), 31)
        np.testing.assert_array_equal(result['barcodes'], query.obs_names)
        np.testing.assert_array_equal(result['proteins'], PANEL)
    # Independent scoring remains possible after all model artifacts disappear.
    shutil.rmtree(bundle)
    protein = ad.AnnData(sparse.csr_matrix(np.random.default_rng(17).poisson(3, (len(query), 31)).astype(np.float32)))
    protein.obs_names, protein.var_names = query.obs_names, PANEL
    protein_path = tmp_path / 'query_adt.h5ad'; protein.write_h5ad(protein_path)
    scoring_registry = tmp_path / 'scoring.json'
    scoring_registry.write_text(json.dumps([{'name': 'query', 'rna': str(query_path), 'adt': str(protein_path)}]))
    monkeypatch.setattr(Cohort, 'measured_protein', original_reader)
    report = score(scoring_registry, prediction, tmp_path / 'score', extended=True)
    assert report['summary']['spots'] == len(query)


def test_source_overlap_is_rejected_before_output(tmp_path):
    refs = [SimpleNamespace(rna=SimpleNamespace(name='a1', group='a')),
            SimpleNamespace(rna=SimpleNamespace(name='b', group='b'))]
    held = [SimpleNamespace(rna=SimpleNamespace(name='a2', group='a'))]
    with pytest.raises(ValueError, match='disjoint validation'):
        training.fit(refs, held, {}, ResidualConfig(n_genes=8), TrainConfig(), tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
