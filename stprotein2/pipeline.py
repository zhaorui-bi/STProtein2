"""Fit the fixed B0 foundation and predict a whole RNA slice."""
import json
from numbers import Integral
from pathlib import Path
import shutil
import tempfile

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .anchor import fit_anchor
from .data import Cohort, PANEL
from .graph import physical_graph
from .io import freeze_run, sha256, verify_marker, write_json
from .provenance import sources, snapshot_sources
from .smoothing import spatial_residual, apply


def read(path):
    return json.loads(Path(path).read_text())


def check_files(folder, checksums):
    folder = Path(folder).resolve()
    for name, digest in checksums.items():
        path = (folder / name).resolve()
        if folder not in path.parents or sha256(path) != digest:
            raise ValueError('Artifact changed or escaped its directory: ' + name)


def train(registry, groups, output, *, query=None, n_genes=4000, threads=2):
    """Fit all declared references, or exclude an entire development query source."""
    if isinstance(n_genes, bool) or not isinstance(n_genes, Integral) or n_genes < 1:
        raise ValueError('n_genes must be a positive integer')
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError('threads must be a positive integer')
    mapping = read(groups)['groups']
    cohort = Cohort(registry, mapping)
    if query is not None and query not in mapping:
        raise ValueError('Unknown query sample')
    names = sorted(n for n in mapping if query is None or mapping[n] != mapping[query])
    if len({mapping[n] for n in names}) < 2:
        raise ValueError('At least two allowed reference sources are required')
    output = Path(output).resolve()
    protocol = {'stage': 'clean_spatial_ridge_foundation', 'reference_names': names,
                'groups': {n: mapping[n] for n in names}, 'n_genes': int(n_genes),
                'spatial_alpha': .5, 'graph_k': 6, 'graph_radius': 2.5, 'proteins': list(PANEL),
                'anchor_arithmetic': 'FP64_multiply_then_FP32',
                'input_sha256': {cohort.specs[n][m]: sha256(cohort.specs[n][m])
                                 for n in names for m in ('rna', 'adt')},
                'query_ADT_read_or_hashed': False}
    freeze_run(output, protocol)
    if (output / 'bundle_complete.json').exists():
        verify_marker(output, 'bundle_complete.json')
        return output
    anchor, cache = fit_anchor(cohort, names, int(n_genes), output / 'anchor_cache', threads)
    if set(cohort.protein_reads) - set(names):
        raise RuntimeError('A held source entered reference training')
    shutil.copyfile(cache / 'anchor.joblib', output / 'anchor.joblib')
    snapshot_sources(output)
    write_json(output / 'bundle_complete.json', {'manifest_sha256': sha256(output / 'manifest.json'),
               'artifacts': {'anchor.joblib': sha256(output / 'anchor.joblib')},
               'protein_reads': cohort.protein_reads, 'query_ADT_read_or_hashed': False})
    return output


def load_bundle(folder):
    folder = Path(folder).resolve()
    verify_marker(folder, 'bundle_complete.json')
    manifest = read(folder / 'manifest.json')
    if manifest['source_sha256'] != sources():
        raise ValueError('Bundle requires its matching clean implementation')
    protocol = manifest['protocol']
    if (protocol['stage'] != 'clean_spatial_ridge_foundation' or protocol['spatial_alpha'] != .5
            or protocol['proteins'] != list(PANEL) or protocol['query_ADT_read_or_hashed']):
        raise ValueError('Bundle does not match the fixed B0 contract')
    anchor = joblib.load(folder / 'anchor.joblib')
    if (anchor['reference_names'] != protocol['reference_names']
            or anchor['transform'].fitted_names_ != protocol['reference_names']
            or not anchor['independently_fitted'] or anchor['residual_training_seen']):
        raise ValueError('Anchor reference identity differs')
    return anchor


def predict_array(anchor, view):
    """Canonical stable anchor arithmetic followed by the existing physical B0."""
    x = np.where(view.availability > 0, view.x, 0).astype(np.float64)
    ridge = anchor['model']
    base = (x @ ridge.coef_.astype(np.float64).T + ridge.intercept_.astype(np.float64)).astype(np.float32)
    graph = physical_graph(view.coordinates, k=6, radius=2.5)
    local = spatial_residual(base, graph)
    return {'prediction': apply(base, local, .5), 'rna_base': base,
            'spatial_delta': local.astype(np.float32), 'barcodes': view.barcodes,
            'proteins': np.asarray(PANEL)}


def predict_file(bundle, rna, output, *, sample='query', threads=2):
    """Prediction accepts one RNA file; there is no measured-protein input."""
    if (not sample or sample in {'.', '..'} or any(c in sample for c in ('/', '\\', '\0'))
            or isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1):
        raise ValueError('A valid sample name and positive thread count are required')
    bundle, rna, output = Path(bundle).resolve(), Path(rna).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh prediction directory')
    if output == bundle or output in bundle.parents or bundle in output.parents:
        raise ValueError('Prediction and bundle directories must be separate')
    anchor = load_bundle(bundle)
    measured = ad.read_h5ad(rna)
    view = anchor['transform'].transform(measured, sample, 'query')
    with threadpool_limits(threads):
        values = predict_array(anchor, view)
    if values['prediction'].shape != (len(measured), 31) or not np.isfinite(values['prediction']).all():
        raise ValueError('Prediction is not a finite named N x 31 matrix')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stprotein2-predict-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'prediction'
        stage.mkdir()
        np.savez_compressed(stage / 'prediction.npz', **values)
        result = ad.AnnData(values['prediction'], obs=pd.DataFrame(index=view.barcodes),
                            var=pd.DataFrame(index=PANEL))
        result.obsm['spatial'] = view.coordinates
        result.write_h5ad(stage / 'prediction.h5ad')
        write_json(stage / 'manifest.json', {'stage': 'clean_B0_RNA_only_prediction', 'sample': sample,
                   'rna_sha256': sha256(rna), 'bundle_complete_sha256': sha256(bundle / 'bundle_complete.json'),
                   'gene_coverage': float(view.availability.mean()), 'measured_protein_access': False,
                   'source_sha256': sources()})
        write_json(stage / 'prediction_complete.json', {'spots': len(measured), 'proteins': 31,
                   'measured_protein_access': False,
                   'files_sha256': {n: sha256(stage / n) for n in ('manifest.json', 'prediction.npz', 'prediction.h5ad')}})
        stage.rename(output)
    return output
