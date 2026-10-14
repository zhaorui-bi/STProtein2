"""Portable inference parity, real file isolation, corruption and input identity."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import anndata as ad
import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge
import torch

from stprotein2.data import Cohort, PANEL, ReferenceTransform
from stprotein2.graph import FrozenRouter
from stprotein2.io import freeze_run, save_torch, sha256, write_json
from stprotein2.model import STProtein2
from stprotein2.trainer import predict
from stprotein2_anchor.model import AnchoredSTProtein2
from stprotein2_smoothing.experiment import predict as smooth_predict
from stprotein2_spatial_anchor.model import SpatialAnchoredSTProtein2
from stprotein2_spatial_anchor.prediction import predict as spatial_predict
from stprotein2_spatial_anchor.config import SpatialAnchorTrainConfig
from stprotein2_deploy.bundle import export_run, load_bundle, predict_file
from stprotein2_balanced_training.config import BalancedTrainConfig
from stprotein2_execution.memory import configure_execution, CheckpointRepresentations
from test_crossfit import cohort_files, configs


def completed_run(path, family):
    data = path / 'source'; data.mkdir()
    cohort, _, _ = cohort_files(data)
    names = ['s0', 's1', 's2']
    transform = ReferenceTransform(6).fit([cohort.rna(n) for n in names], names)
    query = path / 'application.h5ad'; cohort.rna('s6').write_h5ad(query)
    view = transform.transform(ad.read_h5ad(query), 's6', 'application')
    cfg, train = configs(); cfg = replace(cfg, dropout=0)
    folder = path / 'run'; folder.mkdir()
    torch.manual_seed(61)
    if family == 'spatial_baseline':
        x = np.concatenate([transform.transform(cohort.rna(n), n, n).x for n in names])
        ridge = Ridge().fit(x, np.random.default_rng(3).normal(size=(len(x), 31)))
        anchor = {'transform': transform, 'model': ridge}
        prep = {'anchor': anchor, 'transform': transform, 'alpha': .5, 'mode': 'physical',
                'seed': 2025, 'proteins': PANEL, 'reference_names': names}
        arrays = smooth_predict(anchor, view, .5)
        arrays = {k: arrays[k] for k in ('prediction', 'base', 'local_delta', 'barcodes', 'proteins')}
        write_json(folder / 'selection.json', {'alpha': .5})
    else:
        if family == 'balanced_spatial':
            train = BalancedTrainConfig(**train.to_dict(), spatial_alpha=.5, teacher_mode='none')
            train = replace(train, residual_target='in_fit')
            model = SpatialAnchoredSTProtein2(cfg)
            configure_execution(model, train.execution_policy)
        elif family == 'spatial_neural':
            train = SpatialAnchorTrainConfig(**train.to_dict(), spatial_alpha=.5)
            model = SpatialAnchoredSTProtein2(cfg)
        elif family in ('anchored_neural', 'crossfit'):
            model = AnchoredSTProtein2(cfg)
        else:
            model = STProtein2(cfg)
        if hasattr(model, 'anchor'):
            model.anchor.set_coefficients(np.random.default_rng(2).normal(size=(31, 6)), np.zeros(31))
        with torch.no_grad():
            model.state_readout[-1].weight.normal_(0, .1)
            model.niche_readout[-1].weight.normal_(0, .1)
            model.expert_readout.weight.normal_(0, .1)
        gates = {'gamma_state': (np.arange(31) % 2).tolist(),
                 'gamma_local': (np.arange(31) % 3 != 0).astype(int).tolist(), 'proteins': list(PANEL)}
        model.set_gates(gates['gamma_state'], gates['gamma_local'])
        router = FrozenRouter(seed=2025).fit([transform.transform(cohort.rna(n), n, n).x for n in names], names)
        prediction = spatial_predict if family in ('spatial_neural', 'balanced_spatial') else predict
        arrays, _ = prediction(model, view, router, train, 'cpu')
        arrays['proteins'] = np.asarray(PANEL)
        assert np.max(np.abs(arrays['local_delta'])) > 1e-6
        config = train.to_dict()
        if family not in ('crossfit', 'spatial_neural', 'balanced_spatial'):
            config.pop('residual_target')
        if family == 'joint_neural': config.pop('checkpoint_stage')
        prep = {'transform': transform, 'router': router, 'train_config': config}
        saved = {'model_config': cfg.to_dict(), 'state_dict': model.state_dict()}
        if family == 'crossfit': saved['variant'] = 'residual_CF0'
        if family == 'spatial_neural': saved.update(variant='spatial_basis_CF0', spatial_alpha=.5)
        if family == 'balanced_spatial':
            saved.update(variant=train.variant, spatial_alpha=.5,
                         execution_policy=train.execution_policy.to_dict())
        save_torch(folder / 'model.pt', saved); write_json(folder / 'gate_selection.json', gates)
    joblib.dump(prep, folder / 'preprocessing.joblib')
    freeze_run(folder, {'stage': 'test_completed_outer', 'model': cfg.to_dict(), 'refit_names': names})
    dest = folder / 'predictions/s6'; dest.mkdir(parents=True)
    np.savez_compressed(dest / 'prediction.npz', **arrays)
    artifacts = ['manifest.json', 'preprocessing.joblib']
    artifacts += ['selection.json'] if family == 'spatial_baseline' else ['model.pt', 'gate_selection.json']
    write_json(folder / 'prediction_complete.json', {'queries': ['s6'], 'sha256': {'s6': sha256(dest / 'prediction.npz')},
        'query_protein_read_for_prediction': False,
        'deployment_artifacts': {n: sha256(folder / n) for n in artifacts}})
    return folder, query, arrays


@pytest.mark.parametrize('family', ['joint_neural', 'anchored_neural', 'crossfit', 'spatial_neural', 'spatial_baseline', 'balanced_spatial'])
def test_export_portable_RNA_only_prediction_matches_every_model_family(tmp_path, monkeypatch, family):
    torch.set_num_threads(2)
    run, rna, expected = completed_run(tmp_path, family)
    bundle = tmp_path / 'bundle'
    metadata = export_run(run, bundle)
    assert export_run(run, bundle) == metadata
    if family == 'balanced_spatial':
        _, _, restored, restored_cfg = load_bundle(bundle)
        assert isinstance(restored_cfg, BalancedTrainConfig)
        assert restored_cfg.teacher_mode == 'none' and restored_cfg.residual_target == 'in_fit'
        assert isinstance(restored.representations, CheckpointRepresentations)
    # Model prediction must survive deletion of all source training and ADT files.
    shutil.rmtree(tmp_path / 'source'); shutil.rmtree(run)
    monkeypatch.setattr(Cohort, 'measured_protein', lambda *args: pytest.fail('ADT reader accessed'))
    original_reader = ad.read_h5ad
    reads = []
    def only_query(path, *args, **kwargs):
        assert Path(path).resolve() == rna.resolve()
        reads.append(str(path)); return original_reader(path, *args, **kwargs)
    monkeypatch.setattr(ad, 'read_h5ad', only_query)
    output = tmp_path / 'prediction'
    result = predict_file(bundle, rna, output, sample='s6')
    assert result['measured_protein_access'] is False and len(reads) == 1
    with np.load(output / 'prediction.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['prediction'], expected['prediction'])
        np.testing.assert_array_equal(saved['barcodes'], expected['barcodes'])
        np.testing.assert_array_equal(saved['proteins'], PANEL)
    named = original_reader(output / 'prediction.h5ad')
    np.testing.assert_array_equal(named.X, expected['prediction'])
    assert tuple(named.var_names) == PANEL
    if family != 'spatial_baseline':
        np.testing.assert_array_equal(named.layers['local_delta'], expected['local_delta'])
    assert predict_file(bundle, rna, output, sample='s6') == result and len(reads) == 1
    if family != 'spatial_baseline':
        predict_file(bundle, rna, tmp_path / 'partitioned', sample='s6', execution='adaptive')
        with np.load(tmp_path / 'partitioned/prediction.npz') as saved:
            np.testing.assert_allclose(saved['prediction'], expected['prediction'], atol=1e-5, rtol=1e-5)


def test_bundle_corruption_incomplete_export_and_incompatible_resume_rejected(tmp_path):
    run, rna, _ = completed_run(tmp_path, 'spatial_neural')
    marker = run / 'prediction_complete.json'; contents = marker.read_bytes(); marker.unlink()
    with pytest.raises(ValueError, match='completed outer'): export_run(run, tmp_path / 'bundle')
    marker.write_bytes(contents)
    export_run(run, tmp_path / 'bundle')
    out = tmp_path / 'out'; predict_file(tmp_path / 'bundle', rna, out)
    with pytest.raises(RuntimeError, match='different inputs'):
        predict_file(tmp_path / 'bundle', rna, out, sample='changed')
    with (out / 'prediction.npz').open('ab') as f: f.write(b'changed')
    with pytest.raises(RuntimeError, match='changed'): predict_file(tmp_path / 'bundle', rna, out)
    with (tmp_path / 'bundle/model.pt').open('ab') as f: f.write(b'changed')
    with pytest.raises(RuntimeError, match='changed'): load_bundle(tmp_path / 'bundle')


def test_partial_missing_gene_coverage_raw_counts_and_axes(tmp_path):
    run, rna, _ = completed_run(tmp_path, 'spatial_baseline')
    export_run(run, tmp_path / 'bundle')
    original = ad.read_h5ad(rna)
    genes = json.loads((tmp_path / 'bundle/bundle.json').read_text())['genes']
    partial = original[:, [g for g in original.var_names if g != genes[0]]].copy()
    file = tmp_path / 'partial.h5ad'; partial.write_h5ad(file)
    predict_file(tmp_path / 'bundle', file, tmp_path / 'partial_out')
    diag = json.loads((tmp_path / 'partial_out/diagnostics.json').read_text())
    assert diag['measured_model_genes'] == 5 and diag['missing_model_genes'] == [genes[0]]
    for case in ('zero_overlap', 'fractional_counts', 'duplicate_barcodes', 'no_spatial'):
        value = original.copy()
        if case == 'zero_overlap': value.var_names = [f'unknown{i}' for i in range(value.n_vars)]
        if case == 'fractional_counts': value.X.data += .25
        if case == 'duplicate_barcodes': value.obs_names = ['same'] * value.n_obs
        if case == 'no_spatial': del value.obsm['spatial']
        path = tmp_path / f'{case}.h5ad'; value.write_h5ad(path)
        with pytest.raises(ValueError): predict_file(tmp_path / 'bundle', path, tmp_path / f'{case}_out')
        assert not (tmp_path / f'{case}_out').exists()
