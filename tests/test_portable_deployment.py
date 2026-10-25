"""Export contracts and inference after removing all historical inputs.

The small attestation fixture tests format/integrity checks. Scientific replay
against real independently verified checkpoints is recorded separately.
"""
from dataclasses import asdict
import json
import shutil

import anndata as ad
import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from stprotein2.data import PANEL, ReferenceTransform
from stprotein2.deployment import export_clean, export_research
from stprotein2.io import sha256, write_json
from stprotein2.pipeline import load_bundle, load_residual, predict_array, predict_file, train
from stprotein2.portable import FrozenScaler, load_anchor, scaler_arithmetic, write_anchor
from test_clean_core import cohort


def test_scaler_preserves_float32_rounding_and_per_feature_counts():
    rng = np.random.default_rng(412)
    fitted = StandardScaler().fit(rng.normal(size=(17, 8)).astype(np.float32))
    numeric = FrozenScaler(fitted.mean_, fitted.scale_, fitted.var_, np.full(8, 17), scaler_arithmetic(fitted))
    x = rng.normal(size=(13, 8)).astype(np.float32)
    original = x.copy()
    np.testing.assert_array_equal(numeric.transform(x), fitted.transform(x))
    np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError, match='normalization'):
        FrozenScaler(fitted.mean_, np.zeros(8), fitted.var_, 17)
    with pytest.raises(ValueError, match='floating-point'):
        numeric.transform(np.ones((3, 8), dtype=np.int32))
    # Both previously used scaler rules remain explicit portable contracts.
    old = FrozenScaler(fitted.mean_, fitted.scale_, fitted.var_, 17, 'float64_statistics')
    reference = x.copy(); reference -= fitted.mean_; reference /= fitted.scale_
    np.testing.assert_array_equal(old.transform(x), reference)


def test_baseline_export_survives_deleted_training_inputs(tmp_path):
    registry, groups = cohort(tmp_path)
    run, bundle = tmp_path / 'run', tmp_path / 'bundle'
    train(registry, groups, run, query='query', n_genes=8)
    export_clean(run, bundle)
    assert {p.name for p in bundle.iterdir()} == {'manifest.json', 'bundle_complete.json', 'anchor.joblib'}
    query_path = tmp_path / 'query_rna.h5ad'
    expected = predict_array(load_bundle(run), load_bundle(run)['transform'].transform(ad.read_h5ad(query_path), 'query', 'q'))
    shutil.rmtree(run)
    for entry in json.loads(registry.read_text()):
        for modality in ('rna', 'adt'):
            path = tmp_path / entry[modality]
            if path.exists() and path != query_path:
                path.unlink()
    registry.unlink(); groups.unlink()
    predict_file(bundle, query_path, tmp_path / 'prediction')
    with np.load(tmp_path / 'prediction/prediction.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['prediction'], expected['prediction'])


def research_fixture(tmp_path, monkeypatch, head):
    torch = pytest.importorskip('torch')
    from stprotein2 import graph
    from stprotein2.neural import CompactResidual, ResidualConfig

    torch.set_num_threads(2)
    torch.manual_seed(291)
    registry, groups_path = cohort(tmp_path)
    entries = {r['name']: r for r in json.loads(registry.read_text())}
    groups = json.loads(groups_path.read_text())['groups']
    names = ['a', 'b', 'c']
    transform = ReferenceTransform(8).fit([ad.read_h5ad(entries[n]['rna']) for n in names], names)
    cfg = ResidualConfig(n_genes=8, width=8, hidden=6, members=2, residual_width=10,
                         residual_head=head, dropout=.1)
    model = CompactResidual(cfg).eval()
    with torch.no_grad():
        model.readout[-1].weight.normal_(std=.3)
        (model.gamma_state if head == 'own_mlp' else model.gamma_local)[::3] = 0
    state = {}
    for key, value in model.state_dict().items():
        for prefix, replacement in [('encoder.', 'representations.rna.'), ('readout.', 'residual_readout.'), ('gate.', 'residual_gate.')]:
            if key.startswith(prefix):
                key = replacement + key[len(prefix):]
                break
        state[key] = value.clone()
    state.update({'anchor.weight': torch.randn(31, 8), 'anchor.bias': torch.randn(31),
                  'spatial_alpha': torch.tensor(.5), 'direct.weight': torch.randn(6, 31)})
    run, verification = tmp_path / 'historical_run', tmp_path / 'verification'
    run.mkdir(); verification.mkdir()
    torch.save({'model_config': {**asdict(cfg), 'use_state': False, 'use_niche': False, 'local_operator': 'none'},
                'execution_policy': {'weight_modulated_input': True}, 'state_dict': state,
                'spatial_alpha': .5}, run / 'model.pt')
    # The production converter recognizes this historical class name, then
    # discards it before serialization. The runtime has no router class.
    router_type = type('FrozenRouter', (), {'__module__': 'stprotein2.graph'})
    monkeypatch.setattr(graph, 'FrozenRouter', router_type, raising=False)
    joblib.dump({'transform': transform, 'router': router_type(),
                 'train_config': {'execution_mode': 'full', 'graph_k': 6, 'graph_radius': 2.5,
                                  'spatial_alpha': .5, 'seed': 2025, 'lr': .001,
                                  'residual_target': 'in_fit', 'teacher_mode': 'none'}}, run / 'preprocessing.joblib')
    monkeypatch.delattr(graph, 'FrozenRouter')
    write_json(run / 'gate_selection.json', {'proteins': list(PANEL),
               'gamma_state': model.gamma_state.tolist(), 'gamma_local': model.gamma_local.tolist()})
    write_json(run / 'manifest.json', {'protocol': {'groups': groups, 'refit_names': names,
               'query_names': ['query', 'same_query_source'], 'query_labels_for_selection': False}})
    artifacts = {n: sha256(run / n) for n in ('manifest.json', 'model.pt', 'preprocessing.joblib', 'gate_selection.json')}
    write_json(run / 'prediction_complete.json', {'query_protein_read_for_prediction': False,
               'deployment_artifacts': artifacts, 'manifest_sha256': artifacts['manifest.json']})
    report = {'result_folder': str(run), 'protein_reads_during_replay': [],
              'query_protein_used_for_selection': False, 'numerical_evaluation_only': True,
              'script_sha256': 'synthetic-contract-fixture', 'verification_protocol_sha256': 'synthetic-contract-fixture',
              'hashes': {str(run / n): digest for n, digest in artifacts.items()}}
    write_json(verification / 'report.json', report)
    write_json(verification / 'verified_complete.json', {'status': 'passed',
               'report_sha256': sha256(verification / 'report.json'), 'script_sha256': report['script_sha256'],
               'verification_protocol_sha256': report['verification_protocol_sha256']})
    anchor_folder = tmp_path / 'numeric_anchor'; anchor_folder.mkdir()
    write_anchor(anchor_folder, transform, state['anchor.weight'].numpy(), state['anchor.bias'].numpy(), names)
    anchor = load_anchor(anchor_folder)
    query = ad.read_h5ad(entries['query']['rna'])
    # Exercise missing genes and their availability mask at deployment.
    query = query[:, query.var_names[:-1]].copy()
    query_path = tmp_path / 'query_only.h5ad'; query.write_h5ad(query_path)
    expected = predict_array(anchor, transform.transform(query, 'query', 'q'), model)
    return run, verification, query_path, expected


@pytest.mark.parametrize('head', ['own_mlp', 'neighbor_mlp'])
def test_portable_residual_after_deleting_history_and_references(tmp_path, monkeypatch, head):
    run, verification, query, expected = research_fixture(tmp_path, monkeypatch, head)
    bundle = tmp_path / 'bundle'
    export_research(run, verification, bundle)
    assert {p.name for p in bundle.iterdir()} == {'manifest.json', 'bundle_complete.json', 'anchor.npz',
                                                'transform.json', 'residual.pt', 'conversion.json'}
    assert np.max(np.abs(expected['local_raw'])) > 0
    for path in list(tmp_path.iterdir()):
        if path not in {query, bundle}:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    prediction = tmp_path / 'prediction'
    predict_file(bundle, query, prediction)
    with np.load(prediction / 'prediction.npz', allow_pickle=False) as saved:
        for key in ('prediction', 'base', 'rna_base', 'state_delta', 'local_delta', 'local_raw', 'gamma_state', 'gamma_local'):
            np.testing.assert_array_equal(saved[key], expected[key])
    assert not any('router' in n for n in load_residual(bundle).state_dict())
    with (bundle / 'anchor.npz').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='artifact identity changed'):
        load_bundle(bundle)


def test_unbound_report_and_changed_checkpoint_are_rejected(tmp_path, monkeypatch):
    run, verification, _, _ = research_fixture(tmp_path, monkeypatch, 'own_mlp')
    original = (verification / 'report.json').read_bytes()
    with (verification / 'report.json').open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='independent verification'):
        export_research(run, verification, tmp_path / 'invalid_report')
    (verification / 'report.json').write_bytes(original)
    with (run / 'model.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='deployment artifact changed'):
        export_research(run, verification, tmp_path / 'invalid_weights')
    assert not (tmp_path / 'invalid_weights').exists()
