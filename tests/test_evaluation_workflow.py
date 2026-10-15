"""Independent evaluation, complete metric coverage and frozen-report integrity."""
import json
import shutil

import numpy as np
import pytest

from evaluation.cli import main
from evaluation.metrics import evaluate
from evaluation.prediction import score_prediction
from evaluation.spatial import compare
from stprotein2.data import Cohort
from stprotein2.inference import export_run, predict_file
from test_deploy import completed_run


@pytest.fixture
def standalone(tmp_path):
    run, rna, _ = completed_run(tmp_path, 'balanced_spatial')
    export_run(run, tmp_path / 'bundle')
    prediction = tmp_path / 'prediction'
    predict_file(tmp_path / 'bundle', rna, prediction, sample='s6')
    # Evaluation must survive removal of the entire fitted model and all references.
    adt = tmp_path / 'query_adt.h5ad'
    shutil.copyfile(tmp_path / 'source/s6_adt.h5ad', adt)
    shutil.rmtree(tmp_path / 'source')
    shutil.rmtree(run)
    shutil.rmtree(tmp_path / 'bundle')
    registry = tmp_path / 'evaluation_registry.json'
    registry.write_text(json.dumps([{'name': 's6', 'rna': str(rna), 'adt': str(adt)}]))
    return registry, prediction, tmp_path / 'report', rna


def test_standalone_evaluation_has_no_model_dependency_and_reuses_verified_report(standalone, monkeypatch):
    registry, prediction, output, _ = standalone
    original = (prediction / 'prediction.npz').read_bytes()
    report = score_prediction(registry, prediction, output, extended=True, cluster_counts=(3, 6))
    assert report['summary']['spots'] == 12 and report['summary']['proteins'] == 31
    for key in ('macro_protein_rmse', 'mean_pearson_complete_panel',
                'mean_spearman_complete_panel', 'mean_spot_pearson', 'moran_alignment_MAE', 'ari'):
        assert key in report['summary']
    assert (prediction / 'prediction.npz').read_bytes() == original
    monkeypatch.setattr(Cohort, 'measured_protein', lambda *args: pytest.fail('Verified report should be reused'))
    assert score_prediction(registry, prediction, output, extended=True, cluster_counts=(3, 6)) == report
    with (output / 'scores.json').open('a') as stream:
        stream.write('changed')
    with pytest.raises(ValueError, match='artifact identity changed'):
        score_prediction(registry, prediction, output, extended=True, cluster_counts=(3, 6))


def test_evaluation_cli_and_prediction_input_integrity(standalone, capsys):
    registry, prediction, output, rna = standalone
    main(['prediction', '--registry', str(registry), '--prediction', str(prediction), '--output', str(output)])
    assert json.loads(capsys.readouterr().out)['spots'] == 12
    with pytest.raises(RuntimeError, match='Run code/configuration/inputs changed'):
        score_prediction(registry, prediction, output, extended=True)
    with rna.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='RNA differs'):
        score_prediction(registry, prediction, output)


def test_prediction_checksum_is_checked_before_measured_adt_is_opened(standalone, monkeypatch):
    registry, prediction, output, _ = standalone
    monkeypatch.setattr(Cohort, 'measured_protein', lambda *args: pytest.fail('ADT opened before integrity check'))
    with (prediction / 'prediction.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(RuntimeError, match='Artifact missing or changed'):
        score_prediction(registry, prediction, output)
    assert not output.exists()


def test_complete_panel_and_spot_correlations_match_independent_formulas():
    rng = np.random.default_rng(24)
    y = rng.normal(size=(10, 4))
    p = y + rng.normal(scale=.2, size=y.shape)
    result = evaluate(y, p, ('a', 'b', 'c', 'd'))
    assert result['summary']['mean_pearson_complete_panel'] == pytest.approx(
        np.mean([np.corrcoef(y[:, j], p[:, j])[0, 1] for j in range(4)]))
    assert result['summary']['mean_spot_pearson'] == pytest.approx(
        np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(y, p)]))
    assert result['summary']['macro_protein_rmse'] == pytest.approx(np.sqrt(((p - y) ** 2).mean(0)).mean())
    p[:, 0] = 0
    result = evaluate(y, p, ('a', 'b', 'c', 'd'))
    assert result['summary']['mean_pearson_complete_panel'] is None
    assert result['summary']['mean_spearman_complete_panel'] is None
    assert result['summary']['undefined_correlations'] == 1
    p[0] = 0
    assert evaluate(y, p, ('a', 'b', 'c', 'd'))['summary']['mean_spot_pearson'] is None


def test_invalid_metric_inputs_and_comparison_protocols_do_not_create_wins():
    with pytest.raises(ValueError, match='Nonempty'):
        evaluate(np.empty((0, 2)), np.empty((0, 2)), ('a', 'b'))
    with pytest.raises(ValueError, match='graph edges'):
        evaluate(np.ones((2, 2)), np.ones((2, 2)), ('a', 'b'), np.array([[0], [2]]))
    rows = [{'query': 'q', 'seed': 1, 'rmse': 1.}]
    for queries, seeds, keys, tolerance in [
        (['q', 'q'], [1], {'rmse': 'lower'}, 0),
        (['q'], [1, 1], {'rmse': 'lower'}, 0),
        (['q'], [1], {'rmse': 'typo'}, 0),
        (['q'], [1], {'missing': 'lower'}, 0),
        (['q'], [1], {'rmse': 'lower'}, float('nan')),
    ]:
        with pytest.raises(ValueError):
            compare(rows, rows, queries, seeds, keys, tolerance)
