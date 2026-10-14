"""Common-input outer-source baselines with subset-specific frozen preprocessing.

Training is delegated to the previously tested, unchanged method adapters.
This module adds explicit cohort selection, read-only anchor reuse, artifact
integrity, and query-protein exclusion. Scoring is a separate command.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from stprotein2.data import Cohort, PANEL
from stprotein2.io import freeze_run, sha256, write_json
from stprotein2_crossfit.targets import get_anchor, predict_anchor

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mlp", "tabm", "rna0", "dpas", "dgat")
_ADAPTER = None


def read(path):
    return json.loads(Path(path).read_text())


def neural_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        spec = importlib.util.spec_from_file_location("stprotein2_frozen_neural_baseline", ROOT / "scripts/run_neural_baseline.py")
        _ADAPTER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ADAPTER)
    return _ADAPTER


def sources(method):
    if method not in METHODS:
        raise ValueError("Unknown baseline method")
    paths = []
    for package in ("stprotein2", "stprotein2_anchor", "stprotein2_crossfit", "stprotein2_extended_baselines"):
        paths += sorted((ROOT / package).glob("*.py"))
    paths.append(ROOT / "docs/EXTENDED_BASELINES_PROTOCOL_20261013.md")
    if method == "dgat":
        from stprotein2_baselines.dgat import REPO, upstream
        upstream()
        paths += sorted((ROOT / "stprotein2_baselines").glob("*.py"))
        paths += sorted((REPO / "Model").glob("*.py")) + sorted((REPO / "utils").glob("*.py"))
    else:
        paths += [ROOT / "scripts/run_neural_baseline.py", ROOT.parent / "stprotein_journal/models.py",
                  ROOT.parent / "stprotein_journal/dpas_models.py"]
        if method == "tabm":
            paths += [ROOT / "vendor/tabm/tabm.py", ROOT / "vendor/baseline_deps/wheel_provenance.json"]
        if method == "dpas":
            paths += sorted((ROOT.parent / "DPAS-Graph/dpas-graph").rglob("*.py"))
    return {str(p.relative_to(ROOT.parent)): sha256(p) for p in paths}


def snapshot(output, identities):
    for relative, checksum in identities.items():
        source = ROOT.parent / relative
        if sha256(source) != checksum:
            raise RuntimeError("Source changed before snapshot")
        target = output / "source_snapshot" / relative
        if target.exists() and sha256(target) != checksum:
            raise RuntimeError("Frozen baseline source snapshot changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())


def verify_marker(output, filename):
    marker = read(output / filename)
    if sha256(output / "manifest.json") != marker["manifest_sha256"]:
        raise ValueError("Completed manifest identity changed")
    for relative, checksum in marker["artifacts"].items():
        if sha256(output / relative) != checksum:
            raise ValueError("Completed artifact identity changed: " + relative)
    return marker


def prepare(cohort, names, output, args):
    bundle, source = get_anchor(cohort, names, args.genes, args.threads)
    transform = bundle["transform"]
    if transform.fitted_names_ != names or bundle["reference_names"] != names:
        raise ValueError("Preprocessing training identity differs")
    lineage = {"reference_names": names, "reference_groups": sorted({cohort.groups[n] for n in names}),
               "anchor_path": str(source.resolve()), "anchor_sha256": sha256(source / "anchor.joblib"),
               "anchor_manifest_sha256": sha256(source / "manifest.json"), "genes": list(transform.genes_)}
    prep = output / "preprocessing.joblib"
    record = output / "preprocessing_lineage.json"
    if prep.exists() != record.exists():
        # Rebuild an interrupted preprocessing write from the validated source.
        if (output / "manifest.json").exists():
            raise RuntimeError("Frozen preprocessing pair is incomplete")
    if record.exists():
        previous = read(record)
        if previous["lineage"] != lineage or sha256(prep) != previous["preprocessing_sha256"]:
            raise ValueError("Cached preprocessing identity changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        temporary = prep.with_suffix(".tmp")
        joblib.dump({"transform": transform, "reference_names": names}, temporary)
        temporary.replace(prep)
        write_json(record, {"lineage": lineage, "preprocessing_sha256": sha256(prep)})
    return bundle, prep, lineage


def fit_block(cohort, training, validation, output, budget, args):
    train_groups = {cohort.groups[n] for n in training}
    if not training or budget < 1 or train_groups & {cohort.groups[n] for n in validation}:
        raise ValueError("Invalid or overlapping source split")
    if len({cohort.groups[n] for n in validation}) > 1:
        raise ValueError("Epoch selection must hold one complete source group")
    bundle, prep, lineage = prepare(cohort, training, output, args)
    freeze_run(output, {"stage": "matched_extended_baseline_fit", "method": args.method,
        "training": training, "validation": validation, "budget": budget, "seed": args.seed,
        "device": args.device, "threads": args.threads, "preprocessing_lineage": lineage,
        "preprocessing_sha256": sha256(prep), "dependencies": sources(args.method),
        "inputs": {cohort.specs[n][m]: sha256(cohort.specs[n][m]) for n in training + validation for m in ("rna", "adt")}})
    marker = output / "fit_complete.json"
    if marker.exists():
        verify_marker(output, marker.name)
    if args.method == "dgat":
        from stprotein2_baselines.dgat import fit_model
        result = fit_model(cohort, training, validation, prep, output, budget, args)
    else:
        result = neural_adapter().fit_model(cohort, training, validation, prep, args.method, output, budget, args)
    model, report, transform = result
    if transform.fitted_names_ != training or not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Invalid trained model or preprocessing")
    files = ("model.pt", "training.json", "preprocessing.joblib", "preprocessing_lineage.json")
    complete = {"manifest_sha256": sha256(output / "manifest.json"), "training": training, "validation": validation,
                "selected_epoch": report["selected_epoch"], "artifacts": {p: sha256(output / p) for p in files}}
    if marker.exists() and read(marker) != complete:
        raise ValueError("Completed fit changed during reuse")
    write_json(marker, complete)
    return model, report, transform, bundle


def predict(model, view, cache, args):
    if args.method == "dgat":
        from stprotein2_baselines.dgat import predict as dgat_prediction
        return dgat_prediction(model, view, cache, args.device)
    return neural_adapter().prediction(model, args.method, view, cache, args.device, args.seed)


def run(args):
    if args.method not in METHODS or min(args.genes, args.epochs, args.threads) < 1:
        raise ValueError("Invalid method or training settings")
    torch.set_num_threads(args.threads)
    groups = read(args.groups)["groups"]
    if args.query not in groups:
        raise ValueError("Unknown query")
    cohort = Cohort(args.registry, groups)
    queries = [n for n in groups if groups[n] == groups[args.query]]
    references = [n for n in groups if groups[n] != groups[args.query]]
    source_groups = sorted({groups[n] for n in references})
    if len(source_groups) < 3:
        raise ValueError("Need at least three reference sources for independently selected preprocessing")
    cohort.anchor_cache = Path(args.anchor_cache).resolve()
    cohort.reuse_anchor_caches = [Path(p).resolve() for p in args.reuse_anchor_cache]
    cohort.anchor_cache_policy = args.anchor_cache_policy
    protein_reader = cohort.measured_protein
    def reference_protein_only(name):
        if name in queries:
            raise RuntimeError("Outer query protein access forbidden")
        return protein_reader(name)
    cohort.measured_protein = reference_protein_only
    output = Path(args.output).resolve()
    identities = sources(args.method)
    inputs = {cohort.specs[n][m]: sha256(cohort.specs[n][m])
              for n in references for m in ("rna", "adt")}
    inputs.update({cohort.specs[n]["rna"]: sha256(cohort.specs[n]["rna"]) for n in queries})
    freeze_run(output, {"stage": "seen_development_extended_common_input_baseline", "method": args.method,
        "groups": groups, "query_names": queries, "refit_names": references,
        "genes": args.genes, "epochs": args.epochs, "seed": args.seed, "device": args.device, "threads": args.threads,
        "selection": "median reference leave-source-out selected epochs; validation every4; patience6",
        "sampling": "one uniformly sampled slice per reference source per epoch; no augmentation",
        "preprocessing": "exact transform from independently fitted matching-subset frozen anchor; no query RNA fitting",
        "prediction_base": "same-reference frozen RNA anchor, not the neural baseline prediction",
        "query_labels_for_selection": False, "query_ADT_file_opened_or_hashed": False,
        "published_score_reproduction": False, "dependencies": identities, "inputs": inputs,
        "reference_specs": {n: cohort.specs[n] for n in references},
        "query_rna_specs": {n: cohort.specs[n]["rna"] for n in queries}})
    snapshot(output, identities)
    marker = output / "prediction_complete.json"
    if marker.exists():
        complete = verify_marker(output, marker.name)
        if complete["queries"] != queries:
            raise ValueError("Completed query coverage changed")
        for folder in [output / f"inner_{i:02d}" for i in range(len(source_groups))] + [output / "refit"]:
            verify_marker(folder, "fit_complete.json")
        print("ALREADY_COMPLETE " + str(output), flush=True)
        return complete
    with threadpool_limits(args.threads):
        selected = []
        for i, group in enumerate(source_groups):
            training = [n for n in references if groups[n] != group]
            validation = [n for n in references if groups[n] == group]
            print(f"EXTENDED_BASELINE {args.method} {args.query} inner={i+1}/{len(source_groups)} held={group}", flush=True)
            model, report, _, _ = fit_block(cohort, training, validation, output / f"inner_{i:02d}", args.epochs, args)
            selected.append(report["selected_epoch"])
            del model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        epochs = max(1, int(np.median(selected)))
        write_json(output / "epoch_selection.json", {"inner_selected_epochs": selected, "selected_epoch": epochs})
        model, report, transform, bundle = fit_block(cohort, references, [], output / "refit", epochs, args)
        reads = list(cohort.protein_reads)
        hashes = {}
        for query in queries:
            view = cohort.input(query, transform)
            values = predict(model, view, output / "graphs", args)
            base = predict_anchor(bundle, view)
            if values.shape != base.shape or not np.isfinite(values).all():
                raise ValueError("Invalid prediction shape or values")
            destination = output / "predictions" / query / "prediction.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, prediction=values, base=base, barcodes=view.barcodes, proteins=np.asarray(PANEL))
            temporary.replace(destination)
            hashes[query] = sha256(destination)
        if cohort.protein_reads != reads or set(reads) & set(queries):
            raise RuntimeError("Query protein accessed during model selection or prediction")
        artifacts = {"epoch_selection.json": sha256(output / "epoch_selection.json")}
        for folder in [output / f"inner_{i:02d}" for i in range(len(source_groups))] + [output / "refit"]:
            artifacts[str((folder / "fit_complete.json").relative_to(output))] = sha256(folder / "fit_complete.json")
        artifacts.update({f"predictions/{q}/prediction.npz": h for q, h in hashes.items()})
        complete = {"queries": queries, "sha256": hashes, "manifest_sha256": sha256(output / "manifest.json"),
                    "artifacts": artifacts, "reference_names": references, "query_protein_read_for_prediction": False,
                    "protein_reads_before_prediction": reads, "selected_epoch": epochs,
                    "refit_parameters": report["parameters"], "fits": len(source_groups) + 1}
        write_json(marker, complete)
    print("EXTENDED_BASELINE_COMPLETE " + str(output), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("registry", "groups", "query", "output", "anchor-cache"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reuse-anchor-cache", action="append", default=[])
    parser.add_argument("--anchor-cache-policy", choices=("fit", "reuse_frozen"), default="reuse_frozen")
    parser.add_argument("--genes", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
