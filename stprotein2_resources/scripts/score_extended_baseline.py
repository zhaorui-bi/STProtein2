"""Score frozen predictions; only this separate step opens query protein files."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stprotein2.data import Cohort, PANEL
from stprotein2.evaluation import evaluate
from stprotein2.graph import physical_graph
from stprotein2.io import freeze_run, sha256, write_json
from stprotein2_extended_baselines.runner import verify_marker


def score(registry, output):
    output = Path(output).resolve()
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
    freeze_run(output / "scoring", {"stage": "separate_seen_development_query_scoring",
        "prediction_manifest_sha256": sha256(output / "manifest.json"),
        "predictions_sha256": complete["sha256"], "query_inputs_sha256": inputs,
        "script_sha256": sha256(__file__), "query_protein_used_for_model_selection": False})
    summaries, artifacts = {}, {}
    for query, values in predictions.items():
        target = cohort.measured_protein(query)
        graph = physical_graph(cohort.rna(query).obsm["spatial"])
        report = evaluate(target, values["prediction"], PANEL, graph.edge_index)
        report["base_summary"] = evaluate(target, values["base"], PANEL)["summary"]
        report["scope"] = "seen development source holdout; not independent confirmation or a complete SOTA ranking"
        destination = output / "predictions" / query / "scores.json"
        write_json(destination, report)
        summaries[query] = report["summary"]
        artifacts[str(destination.relative_to(output))] = sha256(destination)
    write_json(output / "development_scores.json", summaries)
    artifacts["development_scores.json"] = sha256(output / "development_scores.json")
    write_json(output / "score_complete.json", {"queries": complete["queries"], "sha256": artifacts,
        "scoring_manifest_sha256": sha256(output / "scoring/manifest.json"), "protein_reads": cohort.protein_reads,
        "prediction_sha256": complete["sha256"]})
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    score(args.registry, args.output)
