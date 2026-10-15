"""Evaluate a completed standalone prediction without loading its model or references."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from stprotein2.data import Cohort, PANEL
from stprotein2.graph import physical_graph
from stprotein2.inference.bundle import check_files
from stprotein2.io import freeze_run, sha256, verify_marker, write_json

from .score import evaluation_sources, make_report, metric_settings


def score_prediction(registry, prediction, output, *, extended=False, cluster_counts=(4, 6, 8)):
    """Bind query RNA/ADT and frozen prediction identities to a separate atomic report."""
    prediction, output = Path(prediction).resolve(), Path(output).resolve()
    if output == prediction or output in prediction.parents:
        raise ValueError('Evaluation output must be separate from the prediction directory')
    settings = metric_settings(extended, cluster_counts)
    complete = json.loads((prediction / 'prediction_complete.json').read_text())
    if (complete.get('measured_protein_access') is not False
            or not {'manifest.json', 'prediction.npz'} <= set(complete.get('files_sha256', {}))):
        raise ValueError('A completed standalone RNA-only prediction is required')
    check_files(prediction, complete['files_sha256'])
    identity = json.loads((prediction / 'manifest.json').read_text())
    query = identity['sample']
    cohort = Cohort(registry)
    if query not in cohort.specs:
        raise ValueError('Prediction sample is missing from the evaluation registry')
    spec = cohort.specs[query]
    if sha256(spec['rna']) != identity['rna_sha256']:
        raise ValueError('Evaluation RNA differs from the prediction input')
    with np.load(prediction / 'prediction.npz', allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    rna = cohort.rna(query)
    if (not np.array_equal(values['barcodes'], rna.obs_names.to_numpy())
            or tuple(values['proteins']) != PANEL or complete['spots'] != len(rna)
            or complete['proteins'] != len(PANEL)):
        raise ValueError('Evaluation prediction axes differ')
    protocol = {
        'stage': 'separate_standalone_prediction_evaluation', 'sample': query,
        'prediction_manifest_sha256': sha256(prediction / 'manifest.json'),
        'prediction_complete_sha256': sha256(prediction / 'prediction_complete.json'),
        'prediction_sha256': sha256(prediction / 'prediction.npz'),
        'query_inputs_sha256': {spec[key]: sha256(spec[key]) for key in ('rna', 'adt')},
        'evaluation_source_sha256': evaluation_sources(), 'metric_settings': settings,
        'query_protein_used_for_model_selection': False,
    }
    if output.exists():
        verify_marker(output, 'score_complete.json')
        freeze_run(output, protocol)
        return json.loads((output / 'scores.json').read_text())
    target = cohort.measured_protein(query)
    report = make_report(target, values, physical_graph(rna.obsm['spatial']), settings)
    report['sample'] = query
    report['scope'] = 'completed prediction evaluation; independence depends on the experimental split'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stprotein2-evaluation-', dir=output.parent) as scratch:
        stage = Path(scratch) / 'report'
        freeze_run(stage, protocol)
        write_json(stage / 'scores.json', report)
        write_json(stage / 'score_complete.json', {
            'sample': query, 'manifest_sha256': sha256(stage / 'manifest.json'),
            'artifacts': {'scores.json': sha256(stage / 'scores.json')},
            'protein_reads': cohort.protein_reads,
        })
        stage.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m evaluation prediction', description=__doc__)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--prediction', required=True, help='Completed stprotein2 predict directory')
    parser.add_argument('--output', required=True, help='Separate destination for evaluation reports')
    parser.add_argument('--extended', action='store_true', help='Include spatial and domain metrics')
    parser.add_argument('--cluster-counts', type=int, nargs='+', default=[4, 6, 8])
    args = parser.parse_args(argv)
    report = score_prediction(args.registry, args.prediction, args.output,
                              extended=args.extended, cluster_counts=tuple(args.cluster_counts))
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
