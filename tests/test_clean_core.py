"""Verify leakage boundaries, deployability, stable math and artifact rejection."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from stprotein2.data import PANEL, RNAView
from stprotein2.graph import physical_graph
from stprotein2.pipeline import train, predict_file, predict_array, load_bundle
from stprotein2.smoothing import spatial_residual, apply
from evaluation.__main__ import score


def cohort(tmp_path):
    rng = np.random.default_rng(931)
    entries, groups = [], {}
    for index, name in enumerate(('a', 'b', 'c', 'query', 'same_query_source')):
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (12, 8)).astype(np.float32)))
        rna.obs_names = [f'{name}_{j}' for j in range(12)]
        rna.var_names = [f'g{j}' for j in range(8)]
        rna.obsm['spatial'] = np.array([(j // 4, j % 4) for j in range(12)], dtype=float)
        rna_path, adt_path = tmp_path / f'{name}_rna.h5ad', tmp_path / f'{name}_adt.h5ad'
        rna.write_h5ad(rna_path)
        if index < 3:
            protein = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (12, 31)).astype(np.float32)))
            protein.obs_names, protein.var_names = rna.obs_names, PANEL
            protein.write_h5ad(adt_path)
        entries.append({'name': name, 'rna': str(rna_path), 'adt': str(adt_path)})
        groups[name] = name if index < 3 else 'held_source'
    registry, grouping = tmp_path / 'registry.json', tmp_path / 'groups.json'
    registry.write_text(json.dumps(entries))
    grouping.write_text(json.dumps({'groups': groups}))
    return registry, grouping


def test_real_training_rna_only_deployment_and_model_free_scoring(tmp_path):
    registry, groups = cohort(tmp_path)
    bundle, prediction = tmp_path / 'bundle', tmp_path / 'prediction'
    train(registry, groups, bundle, query='query', n_genes=8)
    trained = load_bundle(bundle)
    assert trained['reference_names'] == ['a', 'b', 'c']
    assert set(json.loads((bundle / 'bundle_complete.json').read_text())['protein_reads']) == {'a', 'b', 'c'}
    train(registry, groups, bundle, query='query', n_genes=8)
    for name in ('a', 'b', 'c'):
        (tmp_path / f'{name}_rna.h5ad').unlink()
        (tmp_path / f'{name}_adt.h5ad').unlink()
    predict_file(bundle, tmp_path / 'query_rna.h5ad', prediction)
    with np.load(prediction / 'prediction.npz') as values:
        assert values['prediction'].shape == (12, 31)
        assert np.isfinite(values['prediction']).all()
        assert tuple(values['proteins']) == PANEL
        saved = values['prediction'].copy()
    assert not (tmp_path / 'query_adt.h5ad').exists()
    raw = ad.read_h5ad(tmp_path / 'query_rna.h5ad')
    protein = ad.AnnData(sparse.csr_matrix(np.random.default_rng(42).poisson(3, (12, 31)).astype(np.float32)))
    protein.obs_names, protein.var_names = raw.obs_names, PANEL
    protein.write_h5ad(tmp_path / 'query_adt.h5ad')
    shutil.rmtree(bundle)
    report = score(registry, prediction, tmp_path / 'report', extended=True)
    assert report['summary']['spots'] == 12
    assert score(registry, prediction, tmp_path / 'report', extended=True) == report
    assert np.array_equal(ad.read_h5ad(prediction / 'prediction.h5ad').X, saved)
    with (prediction / 'prediction.npz').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='Artifact changed'):
        score(registry, prediction, tmp_path / 'other_report')


def test_stable_anchor_and_spatial_identity():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(9, 8)).astype(np.float32)
    availability = np.ones(8, dtype=np.float32)
    availability[2] = 0
    x[:, 2] = 1000
    coefficients = rng.normal(size=(31, 8)).astype(np.float32)
    intercept = rng.normal(size=31).astype(np.float32)
    coordinates = np.array([(j // 3, j % 3) for j in range(9)], dtype=float)
    view = RNAView('q', 'q', x, availability, np.zeros((9, 4), dtype=np.float32),
                   coordinates, np.array([f'b{j}' for j in range(9)]), tuple(f'g{j}' for j in range(8)))
    values = predict_array({'model': SimpleNamespace(coef_=coefficients, intercept_=intercept)}, view)
    masked = x.copy(); masked[:, 2] = 0
    expected = (masked.astype(np.float64) @ coefficients.astype(np.float64).T + intercept.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(values['rna_base'], expected)
    graph = physical_graph(coordinates)
    independent = np.zeros_like(expected, dtype=np.float64)
    for i in range(len(x)):
        neighbors = graph.edge_index[1, graph.edge_index[0] == i]
        if len(neighbors):
            independent[i] = expected[neighbors].astype(np.float64).mean(0) - expected[i]
    valid = graph.degree > 0
    independent[valid] -= independent[valid].mean(0)
    np.testing.assert_array_equal(values['prediction'], (expected.astype(np.float64) + .5 * independent).astype(np.float32))


def test_isolated_spot_and_zero_alpha_preserve_rna_anchor():
    base = np.arange(31, dtype=np.float32)[None, :]
    graph = physical_graph([[0, 0]])
    local = spatial_residual(base, graph)
    np.testing.assert_array_equal(apply(base, local, .5), base)
    np.testing.assert_array_equal(apply(base, local, 0), base)


def test_runtime_has_no_neural_dependency():
    code = '''
import sys
class BlockTorch:
 def find_spec(self, fullname, *args):
  if fullname == 'torch' or fullname.startswith('torch.'):
   raise RuntimeError('Torch must not be imported by the clean baseline')
sys.meta_path.insert(0, BlockTorch())
import stprotein2.pipeline
import evaluation.__main__
assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], check=True)


@pytest.mark.parametrize('genes', [0, True, 1.5])
def test_invalid_gene_counts_fail_before_data_access(tmp_path, genes):
    with pytest.raises(ValueError, match='positive integer'):
        train('absent', 'absent', tmp_path / 'run', n_genes=genes)
