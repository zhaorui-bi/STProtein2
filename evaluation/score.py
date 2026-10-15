"""Score frozen predictions; only this separate step opens query protein files."""
import argparse
import json
from pathlib import Path

import numpy as np

from stprotein2.data import Cohort, PANEL
from evaluation.metrics import evaluate
from stprotein2.graph import physical_graph
from stprotein2.io import freeze_run, sha256, write_json, verify_marker
from stprotein2.inference.bundle import check_files


def evaluation_sources():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob('*.py'))}


def metric_settings(extended=False, cluster_counts=(4, 6, 8)):
    counts = tuple(cluster_counts)
    if not counts or len(set(counts)) != len(counts) or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 2 for k in counts):
        raise ValueError('Distinct integer cluster counts >= 2 required')
    return {'extended': bool(extended), 'cluster_counts': list(counts) if extended else []}


def make_report(target, values, graph, settings):
    report = evaluate(target, values['prediction'], PANEL, graph.edge_index)
    if 'base' in values:
        report['base_summary'] = evaluate(target, values['base'], PANEL)['summary']
    if settings['extended']:
        from evaluation.spatial import extended_metrics
        spatial = extended_metrics(target, values['prediction'], graph, settings['cluster_counts'])
        report['summary'].update(spatial['summary'])
        report['per_protein_spatial'] = spatial['per_protein_spatial']
    report['metric_settings'] = settings
    return report


def score(registry, output, *, extended=False, cluster_counts=(4, 6, 8), report_dir=None):
    settings = metric_settings(extended, cluster_counts)
    output = Path(output).resolve()
    destination_root = Path(report_dir).resolve() if report_dir is not None else output
    complete = verify_marker(output, "prediction_complete.json")
    protocol = json.loads((output / "manifest.json").read_text())["protocol"]
    if complete["queries"] != protocol["query_names"]:
        raise ValueError("Scoring query identity differs")
    cohort = Cohort(registry, protocol["groups"])
    predictions, inputs = {}, {}
    for query in complete["queries"]:
        rna_path = cohort.specs[query]["rna"]
        if sha256(rna_path) != protocol["inputs"].get(rna_path):
            raise ValueError("Scoring query RNA identity changed")
        path = output / "predictions" / query / "prediction.npz"
        if sha256(path) != complete["sha256"][query]:
            raise ValueError("Prediction checksum changed")
        with np.load(path, allow_pickle=False) as archive:
            predictions[query] = {k: archive[k].copy() for k in archive.files}
        values = predictions[query]
        if not np.array_equal(values["barcodes"], cohort.rna(query).obs_names.to_numpy()) or tuple(values["proteins"]) != PANEL:
            raise ValueError("Scoring prediction axes differ")
        for modality in ("rna", "adt"):
            source = cohort.specs[query][modality]
            inputs[source] = sha256(source)
    assert not cohort.protein_reads
    if settings['extended'] and any(
            max(settings['cluster_counts']) >= len(cohort.rna(query)) for query in complete['queries']):
        raise ValueError('Domain cluster counts must be smaller than each query spot count')
    freeze_run(destination_root / "scoring", {"stage": "separate_seen_development_query_scoring",
        "prediction_manifest_sha256": sha256(output / "manifest.json"),
        "predictions_sha256": complete["sha256"], "query_inputs_sha256": inputs,
        "evaluation_source_sha256": evaluation_sources(), "metric_settings": settings,
        "query_protein_used_for_model_selection": False})
    marker = destination_root / 'score_complete.json'
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if (recorded['scoring_manifest_sha256'] != sha256(destination_root / 'scoring/manifest.json')
                or recorded['prediction_sha256'] != complete['sha256']):
            raise ValueError('Completed scoring identity changed')
        check_files(destination_root, recorded['sha256'])
        summaries = json.loads((destination_root / 'development_scores.json').read_text())
        print(json.dumps(summaries, indent=2), flush=True)
        return summaries
    summaries, artifacts = {}, {}
    for query, values in predictions.items():
        target = cohort.measured_protein(query)
        graph = physical_graph(cohort.rna(query).obsm["spatial"])
        report = make_report(target, values, graph, settings)
        report["scope"] = "seen development source holdout; not independent confirmation or a complete SOTA ranking"
        destination = destination_root / "predictions" / query / "scores.json"
        write_json(destination, report)
        summaries[query] = report["summary"]
        artifacts[str(destination.relative_to(destination_root))] = sha256(destination)
    write_json(destination_root / "development_scores.json", summaries)
    artifacts["development_scores.json"] = sha256(destination_root / "development_scores.json")
    write_json(destination_root / "score_complete.json", {"queries": complete["queries"], "sha256": artifacts,
        "scoring_manifest_sha256": sha256(destination_root / "scoring/manifest.json"), "protein_reads": cohort.protein_reads,
        "prediction_sha256": complete["sha256"]})
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m evaluation run', description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument('--report-dir', help='Separate report destination; default: the training run')
    parser.add_argument('--extended', action='store_true', help='Include spatial and domain metrics')
    parser.add_argument('--cluster-counts', type=int, nargs='+', default=[4, 6, 8],
                        help='Prespecified domain cluster counts (with --extended)')
    args = parser.parse_args(argv)
    score(args.registry, args.output, extended=args.extended, cluster_counts=tuple(args.cluster_counts),
          report_dir=args.report_dir)


if __name__ == '__main__':
    main()
