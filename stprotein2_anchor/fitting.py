"""Nested reference-only selection and caching of independent Ridge anchors."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from stprotein2.data import ReferenceTransform
from stprotein2.io import freeze_run, sha256, write_json, source_manifest


LAMBDAS = (1e-4, 1e-3, .01, .1, 1., 10.)


def extra_sources():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def snapshot_sources(output):
    """An immutable local source copy accompanies each real run and cache."""
    root = Path(__file__).resolve().parents[1]
    for package in ("stprotein2", "stprotein2_anchor"):
        destination = Path(output)/"source_snapshot"/package
        destination.mkdir(parents=True, exist_ok=True)
        for source in (root/package).glob("*.py"):
            target = destination/source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise RuntimeError("Frozen source snapshot differs")
            if not target.exists():
                target.write_bytes(source.read_bytes())


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


def load_frozen_anchor(cohort, names, n_genes, cache):
    """Reuse an independently fitted artifact without rewriting its provenance.

    Residual trainer changes do not change an existing Ridge. Its exact data
    subset, target/preprocessing implementation, selection convention, source
    snapshots, environment and artifact checksums must still match. This path
    never fills a missing cache or silently refits using different references.
    """
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Explicit unique anchor references required")
    output = Path(cache)/anchor_key(names, n_genes)
    manifest = json.loads((output/'manifest.json').read_text())
    protocol = manifest['protocol']
    expected = {'stage': 'independent_reference_selected_ridge_anchor',
                'reference_names': names, 'groups': {n: cohort.groups[n] for n in names},
                'n_genes': n_genes, 'lambda_grid': list(LAMBDAS),
                'numpy': np.__version__, 'sklearn': sklearn.__version__,
                'objective': 'equal group / equal slice mean squared error plus lambda times squared weights'}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen anchor protocol is incompatible")
    inputs = {cohort.specs[n][m]: sha256(cohort.specs[n][m]) for n in names for m in ('rna', 'adt')}
    if inputs != protocol['input_sha256']:
        raise ValueError("Frozen anchor input identity changed")
    if source_manifest() != manifest['source_sha256']:
        raise ValueError("Frozen anchor preprocessing/core implementation changed")
    for package, recorded in [('stprotein2', manifest['source_sha256']),
                              ('stprotein2_anchor', protocol['extra_source_sha256'])]:
        snapshot = output/'source_snapshot'/package
        actual = {p.name: sha256(p) for p in snapshot.glob('*.py')}
        if actual != recorded:
            raise ValueError("Frozen anchor source snapshot changed")
    marker = json.loads((output/'complete.json').read_text())
    if marker['reference_names'] != names or sha256(output/'anchor.joblib') != marker['anchor_sha256']:
        raise ValueError("Frozen anchor artifact identity changed")
    bundle = joblib.load(output/'anchor.joblib')
    if bundle['reference_names'] != names or bundle['transform'].fitted_names_ != names or len(bundle['transform'].genes_) != n_genes:
        raise ValueError("Frozen anchor fitted subset changed")
    if bundle['selection'] != marker['selection'] or bundle.get('residual_training_seen') is not False:
        raise ValueError("Frozen anchor selection identity changed")
    return bundle, output


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
