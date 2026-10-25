"""Reference-only grouped Ridge selection with refitted feature transforms."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from .data import ReferenceTransform
from .io import freeze_run,sha256,write_json,source_manifest

LAMBDAS = (1e-4, 1e-3, .01, .1, 1., 10.)

def extra_sources():
    return source_manifest()

def snapshot_sources(output):
    """An immutable local source copy accompanies each real run and cache."""
    from stprotein2.provenance import snapshot_sources as snapshot
    snapshot(output)

def anchor_key(names, n_genes):
    return hashlib.sha256(json.dumps([list(names), n_genes]).encode()).hexdigest()[:24]

def prepare(cohort, names, n_genes):
    transform = ReferenceTransform(n_genes).fit([cohort.rna(n) for n in names], names)
    references = cohort.references(names, transform)
    x = np.concatenate([s.rna.x for s in references])
    y = np.concatenate([s.target for s in references])
    groups = {s.rna.group for s in references}
    weights = np.concatenate([
        np.full(len(s.target), len(x)/(len(groups) *
                sum(r.rna.group == s.rna.group for r in references) * len(s.target)))
        for s in references])
    return transform, x, y, weights

def fit_anchor(cohort, names, n_genes, cache, threads=2):
    """All anchor selection labels belong to names, including inner CV labels.

    Every Ridge CV split refits its own HVGs and standardization. The final
    transform is subsequently shared by the residual model on this subset.
    """
    names = list(names)
    if not names or len(names) != len(set(names)) or not set(names) <= set(cohort.specs):
        raise ValueError("Explicit unique known anchor reference names required")
    groups = sorted({cohort.groups[n] for n in names})
    if len(groups) < 2:
        raise ValueError("Anchor selection requires at least two allowed groups")
    output = Path(cache)/anchor_key(names, n_genes)
    freeze_run(output, {"stage": "independent_reference_selected_ridge_anchor",
               "reference_names": names, "groups": {n: cohort.groups[n] for n in names},
               "n_genes": n_genes, "lambda_grid": LAMBDAS,
               "numpy": np.__version__, "sklearn": sklearn.__version__,
               "objective": "equal group / equal slice mean squared error plus lambda times squared weights",
               "extra_source_sha256": extra_sources(),
               "input_sha256": {cohort.specs[n][m]: sha256(cohort.specs[n][m]) for n in names for m in ("rna", "adt")}})
    snapshot_sources(output)
    bundle_path = output/"anchor.joblib"
    marker = output/"complete.json"
    if marker.exists():
        complete = json.loads(marker.read_text())
        if sha256(bundle_path) != complete["anchor_sha256"]:
            raise ValueError("Anchor cache checksum changed")
        bundle = joblib.load(bundle_path)
        if bundle["reference_names"] != names or bundle["transform"].fitted_names_ != names:
            raise ValueError("Anchor cache reference identity changed")
        return bundle, output
    with threadpool_limits(threads):
        scores = []
        for i, group in enumerate(groups):
            training = [n for n in names if cohort.groups[n] != group]
            validation = [n for n in names if cohort.groups[n] == group]
            path = output/f"inner_{i:02d}.json"
            if path.exists():
                report = json.loads(path.read_text())
                if report["training"] != training or report["validation"] != validation:
                    raise ValueError("Anchor inner-fold identity mismatch")
                scores.append(report["mse"])
                continue
            transform, x, y, weights = prepare(cohort, training, n_genes)
            held = cohort.references(validation, transform)
            losses = []
            for penalty in LAMBDAS:
                ridge = Ridge(alpha=penalty*len(x), solver="cholesky").fit(x, y, sample_weight=weights)
                losses.append(float(np.mean([np.mean((ridge.predict(s.rna.x)-s.target).astype(np.float64)**2)
                                             for s in held])))
            write_json(path, {"training": training, "validation": validation,
                             "preprocessing_fitted_names": transform.fitted_names_, "mse": losses})
            print(f"ANCHOR group={group} references={training} mse={losses}", flush=True)
            scores.append(losses)
        penalty = LAMBDAS[int(np.argmin(np.mean(scores, axis=0)))]
        transform, x, y, weights = prepare(cohort, names, n_genes)
        ridge = Ridge(alpha=penalty*len(x), solver="cholesky").fit(x, y, sample_weight=weights)
        bundle = {"model": ridge, "transform": transform, "reference_names": names,
                  "selection": {"lambda": penalty, "groups": groups, "group_mse": scores},
                  "independently_fitted": True, "residual_training_seen": False}
        temp = bundle_path.with_suffix(".tmp")
        joblib.dump(bundle, temp); temp.replace(bundle_path)
        write_json(marker, {"anchor_sha256": sha256(bundle_path), "reference_names": names,
                            "selection": bundle["selection"]})
    return bundle, output
