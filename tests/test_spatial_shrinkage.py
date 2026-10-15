import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from stprotein2.graph import physical_graph, SpatialGraph
from research.smoothing.operators import apply, permute_graph, select_alpha, spatial_residual


def test_centered_neighbor_residual_preserves_mean_isolation_and_exact_zero_fallback():
    graph = physical_graph(np.array([[0, 0], [1, 0], [2, 0], [100, 0]], dtype=float), k=2, radius=1.1)
    base = np.array([[0., 2.], [2., 1.], [6., 7.], [99., 101.]], dtype=np.float32)
    residual = spatial_residual(base, graph)
    np.testing.assert_allclose(residual[:3, 0], [7/3, 4/3, -11/3], rtol=1e-12)
    np.testing.assert_allclose(residual.sum(0), 0, atol=1e-12)
    np.testing.assert_array_equal(residual[-1], 0)
    np.testing.assert_array_equal(apply(base, residual, 0), base)
    for alpha in [0., .25, 1.]:
        result = apply(base, residual, alpha)
        np.testing.assert_allclose(result.mean(0), base.mean(0), atol=2e-6)
        np.testing.assert_array_equal(result[-1], base[-1])
    with pytest.raises(ValueError):
        apply(base, residual, -1)
    empty = SpatialGraph(4, np.empty((2, 0), dtype=np.int64), graph.coordinates, 1.)
    np.testing.assert_array_equal(spatial_residual(base, empty), np.zeros_like(base))


def test_graph_relabeling_preserves_topology_and_is_stable_under_row_reordering():
    rng = np.random.default_rng(32)
    graph = physical_graph(np.array([(i, j) for i in range(5) for j in range(4)], dtype=float), k=4, radius=1.1)
    keys = np.array([f'barcode_{i:04}' for i in range(graph.n)])
    wrong, permutation = permute_graph(graph, keys, 'slice', seed=2025)
    np.testing.assert_array_equal(np.sort(wrong.degree), np.sort(graph.degree))
    np.testing.assert_array_equal(wrong.adjacency[permutation][:, permutation].toarray(), graph.adjacency.toarray())
    order = rng.permutation(graph.n); inverse = np.argsort(order)
    edge = inverse[graph.edge_index]; edge = edge[:, np.lexsort((edge[1], edge[0]))]
    reordered = SpatialGraph(graph.n, edge, graph.coordinates[order], 1.)
    other, _ = permute_graph(reordered, keys[order], 'slice', seed=2025)
    np.testing.assert_array_equal(other.adjacency[inverse][:, inverse].toarray(), wrong.adjacency.toarray())
    base = rng.normal(size=(graph.n, 3)).astype(np.float32)
    np.testing.assert_allclose(spatial_residual(base[order], other)[inverse], spatial_residual(base, wrong), atol=1e-12)


def test_reference_selection_weights_sources_and_slices_not_spot_counts():
    records = []
    for i in range(10):
        n = 1 if i < 9 else 1000
        records.append({'name': f'a{i}', 'group': 'a', 'base': np.zeros((n, 2), dtype=np.float32),
                        'residual': np.ones((n, 2)), 'target': np.ones((n, 2))})
    records.append({'name': 'b', 'group': 'b', 'base': np.zeros((1, 2), dtype=np.float32),
                    'residual': np.ones((1, 2)), 'target': np.full((1, 2), -2.)})
    result = select_alpha(records)
    assert result['alpha'] == 0
    assert result['mean_group_mse'][0] == 2.5
    for row in records:
        row['residual'] *= 0
    assert select_alpha(records)['alpha'] == 0


def test_nested_spatial_control_excludes_query_and_held_anchor_sources_and_recovers(tmp_path, monkeypatch):
    import argparse
    from stprotein2.data import Cohort
    from research.joint.cli import score_run
    from stprotein2.training.anchors import fit_anchor
    import research.smoothing.experiment as experiment
    from test_crossfit import cohort_files

    _, registry_all, groups_all = cohort_files(tmp_path)
    names = ['s0', 's1', 's2', 's3', 's6']
    entries = [r for r in json.loads(registry_all.read_text()) if r['name'] in names]
    registry = tmp_path / 'four_groups.json'; registry.write_text(json.dumps(entries))
    groups = {n: groups_all[n] for n in names}
    cohort = Cohort(registry, groups)
    cohort.anchor_cache = tmp_path / 'independent_anchors'
    references = names[:-1]
    for held in [None, *sorted({groups[n] for n in references})]:
        allowed = [n for n in references if groups[n] != held]
        fit_anchor(cohort, allowed, 6, cohort.anchor_cache)
    cohort.anchor_cache_policy = 'reuse_frozen'
    cohort.protein_reads.clear()
    real_protein = cohort.measured_protein
    def guard(name):
        assert name != 's6'
        return real_protein(name)
    monkeypatch.setattr(cohort, 'measured_protein', guard)
    out = tmp_path / 'run'
    complete = experiment.run_query(cohort, 's6', out, genes=6)
    assert set(complete) == {'physical', 'permuted'}
    for mode in complete:
        selection = json.loads((out / mode / 'selection.json').read_text())
        assert groups['s6'] not in selection['selection_groups']
        for row in selection['OOF_provenance']:
            anchor = row['anchor']
            assert row['held_source'] not in anchor['reference_groups']
            assert groups['s6'] not in anchor['reference_groups']
            for fold in anchor['CV']:
                assert not {groups['s6'], row['held_source']} & {groups[n] for n in fold['training'] + fold['validation']}
        exported = joblib.load(out / mode / 'preprocessing.joblib')
        adt = Path(cohort.specs['s6']['adt']); hidden = adt.with_suffix('.unavailable')
        adt.rename(hidden)
        try:
            view = cohort.input('s6', exported['transform'])
            prediction = experiment.predict(exported['anchor'], view, exported['alpha'], mode, exported['seed'])
        finally:
            hidden.rename(adt)
        with np.load(out / mode / 'predictions/s6/prediction.npz') as saved:
            np.testing.assert_array_equal(prediction['prediction'], saved['prediction'])
        score_run(argparse.Namespace(registry=str(registry), output=str(out / mode)))
    before = list(cohort.protein_reads)
    assert experiment.run_query(cohort, 's6', out, genes=6) == complete
    assert cohort.protein_reads == before
    (out / 'physical/prediction_complete.json').unlink()
    assert experiment.run_query(cohort, 's6', out, genes=6)['physical']['sha256'] == complete['physical']['sha256']
    with (out / 'physical/predictions/s6/prediction.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(RuntimeError, match='artifact changed'):
        experiment.run_query(cohort, 's6', out, genes=6)
