"""Deployment does not need a cohort registry, training files, or measured ADT.

Bundles contain trusted Python preprocessing objects. Checksums detect changes;
they do not authenticate a model obtained from an untrusted third party.
"""
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from stprotein2.config import ModelConfig, TrainConfig, from_dict
from stprotein2.data import PANEL
from stprotein2.io import sha256, write_json
from stprotein2.models.residual import STProtein2
from stprotein2.training.basic import predict as predict_neural
from stprotein2.training.config import AnchorTrainConfig
from stprotein2.models.anchor import AnchoredSTProtein2
from stprotein2.training.config import ResidualTrainConfig
from stprotein2.inference.baseline import predict as predict_spatial
from stprotein2.training.config import SpatialAnchorTrainConfig
from stprotein2.models.spatial import SpatialAnchoredSTProtein2
from stprotein2.training.prediction import predict as predict_spatial_neural
from stprotein2.training.config import BalancedTrainConfig
from stprotein2.execution.memory import ExecutionPolicy, configure_execution


from stprotein2.paths import SOURCE_ROOT as ROOT
from stprotein2.provenance import sources


def read(path):
    return json.loads(Path(path).read_text())


def runtime():
    return {"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "torch", "anndata", "joblib")}}


def check_files(root, checksums):
    root = Path(root).resolve()
    for relative, digest in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Artifact missing or changed: {relative}")


def check_sources(record):
    for relative, digest in record.items():
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Frozen model implementation differs: {relative}")


def neural_model(saved, device):
    cfg = from_dict(ModelConfig, saved["model_config"])
    if "spatial_alpha" in saved:
        family = "spatial_neural"
        cls = BalancedTrainConfig if saved.get("variant", "").startswith("balanced_") else SpatialAnchorTrainConfig
        model = SpatialAnchoredSTProtein2(cfg, saved["spatial_alpha"])
    elif "anchor.weight" in saved["state_dict"]:
        family = "anchored_neural"
        cls = ResidualTrainConfig if saved.get("variant", "").startswith("residual_") else AnchorTrainConfig
        model = AnchoredSTProtein2(cfg)
    else:
        family, cls, model = "joint_neural", TrainConfig, STProtein2(cfg)
    if "execution_policy" in saved:
        configure_execution(model, ExecutionPolicy(**saved["execution_policy"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Non-finite model checkpoint")
    if family == "spatial_neural" and float(model.spatial_alpha) != saved["spatial_alpha"]:
        raise ValueError("Spatial coefficient metadata disagrees with weights")
    return family, model.to(device).eval(), cls


def export_run(run, output):
    """Export a completed outer run; inner fits and incomplete runs are rejected."""
    run, output = Path(run).resolve(), Path(output).resolve()
    marker_path = run / "prediction_complete.json"
    if not marker_path.is_file():
        raise ValueError("Export requires a completed outer prediction run")
    marker, manifest = read(marker_path), read(run / "manifest.json")
    if marker.get("query_protein_read_for_prediction") is not False:
        raise ValueError("Source run does not declare RNA-only prediction")
    if not marker.get("queries") or set(marker["queries"]) != set(marker["sha256"]):
        raise ValueError("Incomplete source prediction coverage")
    check_files(run, {f"predictions/{name}/prediction.npz": digest for name, digest in marker["sha256"].items()})
    for key in ("deployment_artifacts", "artifacts_sha256"):
        check_files(run, marker.get(key, {}))
    protocol = manifest["protocol"]
    historical = dict(manifest["source_sha256"])
    historical.update(protocol.get("extra_source_sha256", {}))
    historical.update(protocol.get("research_source_sha256", {}))
    for key in ("dependency_source_sha256", "dependency_sources"):
        historical.update(protocol.get(key, {}))
        historical.update(protocol.get("source_shrinkage_protocol", {}).get(key, {}))
    check_sources(historical)
    names = ["manifest.json", "prediction_complete.json", "preprocessing.joblib"]
    names += [n for n in ("model.pt", "gate_selection.json", "selection.json") if (run / n).exists()]
    origin = {n: sha256(run / n) for n in names}
    if output.exists():
        metadata, _, _, _ = load_bundle(output)
        if metadata["origin_artifacts_sha256"] != origin:
            raise RuntimeError("Export destination belongs to a different source run")
        return metadata
    prep = joblib.load(run / "preprocessing.joblib")
    genes = tuple(prep["transform"].genes_)
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("Invalid frozen gene vocabulary")
    if "anchor" in prep:
        if prep.get("mode") != "physical" or tuple(prep.get("proteins", ())) != PANEL:
            raise ValueError("Only physical-graph fixed-panel spatial baselines deploy")
        if tuple(prep["anchor"]["transform"].genes_) != genes:
            raise ValueError("Spatial baseline and preprocessing vocabularies disagree")
        family = "spatial_baseline"
        # Preserve only inference objects, not training data paths or CV records.
        packaged = {"transform": prep["transform"], "anchor": {k: prep["anchor"][k] for k in ("transform", "model")},
                    "alpha": float(prep["alpha"]), "mode": "physical", "seed": int(prep["seed"])}
        if not np.isfinite(packaged["alpha"]) or not 0 <= packaged["alpha"] <= 1:
            raise ValueError("Invalid frozen spatial coefficient")
        model_config, graph = None, {"k": 6, "radius": 2.5}
    else:
        saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        family, model, cls = neural_model(saved, "cpu")
        cfg = from_dict(cls, prep["train_config"])
        gates = read(run / "gate_selection.json")
        if tuple(gates["proteins"]) != PANEL or model.cfg.n_genes != len(genes) or model.cfg.n_proteins != len(PANEL):
            raise ValueError("Checkpoint feature axes disagree")
        for key in ("gamma_state", "gamma_local"):
            if not np.array_equal(getattr(model, key).numpy(), np.asarray(gates[key])):
                raise ValueError("Checkpoint gates disagree with selection")
        if family == "spatial_neural" and float(model.spatial_alpha) != cfg.spatial_alpha:
            raise ValueError("Preprocessing and checkpoint spatial coefficients disagree")
        if model.cfg != from_dict(ModelConfig, protocol["model"]):
            raise ValueError("Checkpoint architecture disagrees with run manifest")
        packaged = {k: prep[k] for k in ("transform", "router", "train_config")}
        model_config, graph = model.cfg.to_dict(), {"k": cfg.graph_k, "radius": cfg.graph_radius}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stprotein2-export-", dir=output.parent) as scratch:
        staging = Path(scratch) / "bundle"; staging.mkdir()
        joblib.dump(packaged, staging / "preprocessing.joblib")
        if family != "spatial_baseline":
            shutil.copyfile(run / "model.pt", staging / "model.pt")
            shutil.copyfile(run / "gate_selection.json", staging / "gate_selection.json")
        for relative in sources():
            dest = staging / "source_snapshot" / relative; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
        metadata = {"schema": 1, "family": family, "genes": list(genes), "proteins": list(PANEL),
                    "model_config": model_config, "graph": graph, "input": "raw RNA counts, gene IDs, unique barcodes, obsm['spatial']; one complete slice",
                    "target": "fixed 31-channel row-centered log1p ADT; signed predictions",
                    "reference_names": list(prep["transform"].fitted_names_),
                    "origin_artifacts_sha256": origin, "origin_stage": protocol.get("stage"),
                    "origin_artifacts_bound_at_completion": "deployment_artifacts" in marker or "artifacts_sha256" in marker,
                    "origin_note": "Older runs without deployment hashes are sealed here at export time; this does not retroactively establish checkpoint provenance",
                    "research_status": "development model; no SOTA or independent-validation claim",
                    "source_sha256": sources(), "runtime": runtime(),
                    "files_sha256": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()}}
        write_json(staging / "bundle.json", metadata)
        staging.rename(output)
    return metadata


def load_bundle(path, device="cpu"):
    path = Path(path).resolve(); metadata = read(path / "bundle.json")
    if metadata.get("schema") != 1 or tuple(metadata["proteins"]) != PANEL:
        raise ValueError("Unsupported deployment bundle schema or protein panel")
    check_files(path, metadata["files_sha256"])
    check_sources(metadata["source_sha256"])
    prep = joblib.load(path / "preprocessing.joblib")
    if tuple(prep["transform"].genes_) != tuple(metadata["genes"]):
        raise ValueError("Bundle gene order differs from preprocessing")
    if metadata["family"] == "spatial_baseline":
        return metadata, prep, None, None
    family, model, cls = neural_model(torch.load(path / "model.pt", map_location="cpu", weights_only=True), device)
    if family != metadata["family"] or model.cfg.to_dict() != metadata["model_config"]:
        raise ValueError("Bundle architecture differs from model")
    return metadata, prep, model, from_dict(cls, prep["train_config"])


def predict_file(bundle, rna_path, output, *, sample="query", device="cpu", threads=2, execution="saved"):
    """Predict one complete slice. No ADT path or cohort registry is accepted."""
    if threads < 1 or execution not in ("saved", "full", "adaptive") or not sample:
        raise ValueError("Invalid prediction options")
    output, rna_path, bundle = Path(output).resolve(), Path(rna_path).resolve(), Path(bundle).resolve()
    with threadpool_limits(threads):
        torch.set_num_threads(threads)
        metadata, prep, model, cfg = load_bundle(bundle, device)
        if model is None and (device != "cpu" or execution == "adaptive"):
            raise ValueError("Simple spatial baseline uses full CPU inference")
        identity = {"bundle_sha256": sha256(bundle / "bundle.json"), "rna_sha256": sha256(rna_path),
                    "sample": sample, "device": device, "threads": threads, "execution": execution,
                    "runtime": runtime(), "source_sha256": sources()}
        if output.exists():
            if not (output / "prediction_complete.json").exists() or read(output / "manifest.json") != identity:
                raise RuntimeError("Prediction output exists with incomplete or different inputs")
            result = read(output / "prediction_complete.json"); check_files(output, result["files_sha256"])
            return result
        rna = ad.read_h5ad(rna_path)
        if rna.n_obs == 0 or "spatial" not in rna.obsm:
            raise ValueError("A nonempty RNA slice with obsm['spatial'] is required")
        if "feature_types" in rna.var and not np.all(rna.var["feature_types"].astype(str) == "Gene Expression"):
            raise ValueError("Input must contain RNA gene features only")
        view = prep["transform"].transform(rna, sample, "application")
        if not np.any(view.availability):
            raise ValueError("No measured input genes overlap the frozen model vocabulary")
        if model is None:
            raw = predict_spatial(prep["anchor"], view, prep["alpha"], "physical", prep["seed"])
            arrays = {k: raw[k] for k in ("prediction", "barcodes", "proteins")}
            arrays.update(base=raw["prediction"], rna_base=raw["base"], spatial_basis_delta=raw["local_delta"],
                          spatial_alpha=np.asarray(prep["alpha"]), state_delta=np.zeros_like(raw["base"]),
                          local_delta=np.zeros_like(raw["base"]))
            diagnostics = {"graph_sha256": raw["graph_sha256"], "isolated_nodes": raw["isolated_nodes"]}
        else:
            cfg = replace(cfg, threads=threads, execution_mode=cfg.execution_mode if execution == "saved" else execution)
            fn = predict_spatial_neural if metadata["family"] == "spatial_neural" else predict_neural
            arrays, diagnostics = fn(model, view, prep["router"], cfg, device)
            arrays["proteins"] = np.asarray(PANEL)
        if arrays["prediction"].shape != (len(rna), len(PANEL)) or not np.isfinite(arrays["prediction"]).all():
            raise RuntimeError("Invalid prediction output")
        diagnostics.update(measured_model_genes=int(view.availability.sum()), model_genes=len(view.genes),
                           gene_coverage_fraction=float(view.availability.mean()),
                           missing_model_genes=[g for g, available in zip(view.genes, view.availability) if not available],
                           measured_protein_access=False, inference_unit="whole_slice", family=metadata["family"],
                           source_runtime=metadata["runtime"], runtime=identity["runtime"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".stprotein2-predict-", dir=output.parent) as scratch:
            stage = Path(scratch) / "result"; stage.mkdir()
            np.savez_compressed(stage / "prediction.npz", **arrays)
            result_adata = ad.AnnData(arrays["prediction"], obs=pd.DataFrame(index=pd.Index(view.barcodes, name="barcode")),
                                     var=pd.DataFrame(index=pd.Index(PANEL, name="protein")))
            result_adata.obsm["spatial"] = view.coordinates.copy()
            for key in ("base", "rna_base", "state_delta", "local_delta", "spatial_basis_delta"):
                if key in arrays: result_adata.layers[key] = arrays[key]
            result_adata.uns["stprotein2"] = {"bundle_sha256": identity["bundle_sha256"], "target": metadata["target"],
                                             "measured_protein_access": False, "research_status": metadata["research_status"]}
            result_adata.write_h5ad(stage / "prediction.h5ad")
            write_json(stage / "manifest.json", identity); write_json(stage / "diagnostics.json", diagnostics)
            result = {"spots": len(rna), "proteins": len(PANEL), "measured_protein_access": False,
                      "files_sha256": {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}}
            write_json(stage / "prediction_complete.json", result)
            stage.rename(output)
        return result
