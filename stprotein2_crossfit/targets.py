"""Label-free baseline predictions with complete, subset-specific lineage."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from stprotein2.data import PANEL
from stprotein2.graph import digest_array
from stprotein2.io import sha256
from stprotein2_anchor.fitting import anchor_key, fit_anchor, load_frozen_anchor
from stprotein2_anchor.model import FrozenRidge


def get_anchor(cohort, names, n_genes, threads=2):
    """Completed shared artifacts are read-only; writes have a separate cache.

    A cache with no complete marker is not treated as terminal or overwritten.
    The caller controls scheduling if another preparation process is live.
    """
    caches = [Path(cohort.anchor_cache)] + [Path(p) for p in getattr(cohort, "reuse_anchor_caches", ())]
    for cache in caches:
        if (cache/anchor_key(names, n_genes)/"complete.json").exists():
            return load_frozen_anchor(cohort, names, n_genes, cache)
    if getattr(cohort, "anchor_cache_policy", "fit") == "reuse_frozen":
        raise FileNotFoundError(f"No completed independent anchor for {list(names)}")
    return fit_anchor(cohort, names, n_genes, cohort.anchor_cache, threads)


def predict_anchor(bundle, view):
    """Same FP32 FrozenRidge arithmetic as deployment, without protein reads."""
    if tuple(bundle["transform"].genes_) != tuple(view.genes):
        raise ValueError("Baseline preprocessing gene identity mismatch")
    layer = FrozenRidge(len(view.genes), len(PANEL))
    layer.set_coefficients(bundle["model"].coef_, bundle["model"].intercept_)
    with torch.no_grad():
        return layer(torch.as_tensor(view.x), torch.as_tensor(view.availability)).numpy().copy()


@dataclass(frozen=True)
class BaselineTarget:
    name: str
    group: str
    barcodes: tuple
    prediction: np.ndarray
    provenance: dict

    def __post_init__(self):
        prediction = np.array(self.prediction, dtype=np.float32, copy=True)
        if prediction.shape != (len(self.barcodes), len(PANEL)) or not np.isfinite(prediction).all():
            raise ValueError("Invalid baseline target values")
        if len(set(self.barcodes)) != len(self.barcodes):
            raise ValueError("Duplicate baseline target barcodes")
        prediction.setflags(write=False)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "provenance", json.loads(json.dumps(self.provenance)))

    def signature(self):
        return {"name": self.name, "group": self.group, "barcodes": list(self.barcodes),
                "prediction_sha256": digest_array(self.prediction), "provenance": self.provenance}


def build_targets(cohort, names, n_genes, mode, full_anchor, threads=2):
    """Each OOF anchor excludes the entire target source at every fitting step.

    The only target-side data accessed here is RNA. True measured proteins are
    read separately by the supervised Student training pipeline.
    """
    names = list(names)
    groups = sorted({cohort.groups[n] for n in names})
    if len(names) != len(set(names)) or not names:
        raise ValueError("Unique residual training names required")
    if mode not in {"in_fit", "cross_fitted"}:
        raise ValueError("Unknown residual target protocol")
    if mode == "cross_fitted" and len(groups) < 3:
        raise ValueError("Cross-fitted training requires at least three allowed source groups")
    if full_anchor[0]["reference_names"] != names:
        raise ValueError("Deployment anchor training subset mismatch")
    result = {}
    for group in groups:
        allowed = [n for n in names if mode == "in_fit" or cohort.groups[n] != group]
        bundle, path = full_anchor if mode == "in_fit" else get_anchor(cohort, allowed, n_genes, threads)
        if bundle["reference_names"] != allowed or bundle["transform"].fitted_names_ != allowed:
            raise ValueError("Baseline target anchor training identity mismatch")
        fitted_groups = sorted({cohort.groups[n] for n in allowed})
        if mode == "cross_fitted" and group in fitted_groups:
            raise ValueError("Residual target source leaked into its anchor")
        # Verify the persisted CV lineage, not just the final coefficient fit.
        folds = []
        for record in sorted(Path(path).glob("inner_*.json")):
            row = json.loads(record.read_text())
            if (set(row["training"] + row["validation"]) != set(allowed)
                    or row["preprocessing_fitted_names"] != row["training"]
                    or {cohort.groups[n] for n in row["training"]} & {cohort.groups[n] for n in row["validation"]}):
                raise ValueError("Baseline anchor CV lineage is incompatible")
            folds.append({"file": record.name, "sha256": sha256(record),
                          "training": row["training"], "validation": row["validation"]})
        if len(folds) != len(fitted_groups):
            raise ValueError("Baseline anchor CV lineage is incomplete")
        for name in names:
            if cohort.groups[name] != group:
                continue
            reads = list(cohort.protein_reads)
            view = cohort.input(name, bundle["transform"])
            prediction = predict_anchor(bundle, view)
            if reads != cohort.protein_reads:
                raise RuntimeError("Baseline target prediction read measured proteins")
            lineage = {"mode": mode, "anchor_path": str(Path(path).resolve()),
                       "anchor_sha256": sha256(Path(path)/"anchor.joblib"),
                       "anchor_reference_names": allowed, "anchor_reference_groups": fitted_groups,
                       "anchor_cv": folds, "anchor_genes": list(bundle["transform"].genes_),
                       "anchor_input_fingerprint": view.fingerprint(),
                       "target_protein_read_for_baseline_prediction": False}
            result[name] = BaselineTarget(name, group, tuple(view.barcodes), prediction, lineage)
    return result


def validate_targets(references, baselines, mode):
    if set(baselines) != {s.rna.name for s in references}:
        raise ValueError("Baseline target coverage mismatch")
    allowed = {s.rna.name for s in references}
    groups = {s.rna.name: s.rna.group for s in references}
    for sample in references:
        view = sample.rna
        baseline = baselines[view.name]
        expected = {n for n in allowed if mode == "in_fit" or groups[n] != view.group}
        if (baseline.name != view.name or baseline.group != view.group
                or baseline.barcodes != tuple(view.barcodes)
                or baseline.prediction.shape != sample.target.shape):
            raise ValueError("Baseline target row identity mismatch")
        lineage = baseline.provenance
        if (lineage["mode"] != mode or set(lineage["anchor_reference_names"]) != expected
                or set(lineage["anchor_reference_groups"]) != {groups[n] for n in expected}):
            raise ValueError("Baseline target fitting subset mismatch")
    return {s.rna.name: baselines[s.rna.name].signature() for s in references}
