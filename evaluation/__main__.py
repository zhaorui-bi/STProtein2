"""Score a frozen RNA-only prediction separately from training and inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from stprotein2.data import Cohort, PANEL
from stprotein2.graph import physical_graph
from stprotein2.io import freeze_run, sha256, verify_marker, write_json
from stprotein2.pipeline import check_files
from .metrics import evaluate
from .spatial import extended_metrics


def score(registry, prediction, output, *, extended=False):
    prediction, output = Path(prediction).resolve(), Path(output).resolve()
    if output == prediction or output in prediction.parents or prediction in output.parents:
        raise ValueError('Evaluation requires a separate report directory')
    complete = json.loads((prediction / 'prediction_complete.json').read_text())
    if complete['measured_protein_access'] is not False:
        raise ValueError('Prediction was not RNA-only')
    check_files(prediction, complete['files_sha256'])
    identity = json.loads((prediction / 'manifest.json').read_text())
    cohort = Cohort(registry)
    sample = identity['sample']
    if sha256(cohort.specs[sample]['rna']) != identity['rna_sha256']:
        raise ValueError('Scoring RNA differs from the frozen prediction')
    with np.load(prediction / 'prediction.npz', allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive.files}
    rna = cohort.rna(sample)
    if not np.array_equal(values['barcodes'], rna.obs_names.to_numpy()) or tuple(values['proteins']) != PANEL:
        raise ValueError('Prediction barcode/protein axes differ')
    freeze_run(output, {'stage': 'independent_clean_B0_scoring', 'sample': sample, 'extended': extended,
        'prediction_complete_sha256': sha256(prediction / 'prediction_complete.json'),
        'ADT_sha256': sha256(cohort.specs[sample]['adt']),
        'evaluation_sources': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})
    if (output / 'score_complete.json').exists():
        verify_marker(output, 'score_complete.json')
        return json.loads((output / 'scores.json').read_text())
    truth = cohort.measured_protein(sample)
    graph = physical_graph(rna.obsm['spatial'])
    report = evaluate(truth, values['prediction'], PANEL, graph.edge_index)
    if extended:
        report['spatial'] = extended_metrics(truth, values['prediction'], graph, cluster_counts=(4, 6, 8))
    write_json(output / 'scores.json', report)
    write_json(output / 'score_complete.json', {'manifest_sha256': sha256(output / 'manifest.json'),
               'artifacts': {'scores.json': sha256(output / 'scores.json')}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    print(json.dumps(score(args.registry, args.prediction, args.output, extended=args.extended)['summary'], indent=2))


if __name__ == '__main__':
    main()
