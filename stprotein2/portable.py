"""Portable numeric preprocessing and frozen Ridge; no historical classes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import ReferenceTransform
from .io import write_json


class FrozenScaler:
    """The fitted StandardScaler transform used by the canonical float32 input."""
    def __init__(self, mean, scale, variance, sample_count, arithmetic='input_dtype_statistics'):
        self.mean_, self.scale_, self.var_ = [np.asarray(a, dtype=np.float64).copy() for a in (mean, scale, variance)]
        self.n_features_in_ = len(self.mean_)
        samples = np.asarray(sample_count)
        if (self.mean_.ndim != 1 or self.scale_.shape != self.mean_.shape or self.var_.shape != self.mean_.shape
                or not all(np.isfinite(a).all() for a in (self.mean_, self.scale_, self.var_, samples))
                or np.any(self.scale_ <= 0) or np.any(self.var_ < 0) or np.any(samples <= 0)
                or samples.shape not in {(), self.mean_.shape}):
            raise ValueError('Invalid fitted normalization arrays')
        self.n_samples_seen_ = samples.item() if samples.ndim == 0 else samples.copy()
        if arithmetic not in {'input_dtype_statistics', 'float64_statistics'}:
            raise ValueError('Unknown normalization arithmetic')
        self.arithmetic = arithmetic

    def transform(self, x):
        values = np.asarray(x)
        if (values.ndim != 2 or values.shape[1] != self.n_features_in_
                or values.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(values).all()):
            raise ValueError('Finite aligned floating-point features are required')
        values = values.copy()
        # StandardScaler versions differ in whether statistics are cast before
        # the ufunc. Freeze the actual export-time transform, including rounding.
        dtype = values.dtype if self.arithmetic == 'input_dtype_statistics' else np.float64
        values -= self.mean_.astype(dtype, copy=False)
        values /= self.scale_.astype(dtype, copy=False)
        return values


def scaler_arithmetic(scaler):
    """Identify supported StandardScaler arithmetic without a version guess."""
    probe = np.linspace(-7.1, 9.4, 8 * len(scaler.mean_), dtype=np.float32).reshape(8, -1)
    expected = scaler.transform(probe.copy())
    for rule in ('input_dtype_statistics', 'float64_statistics'):
        numeric = FrozenScaler(scaler.mean_, scaler.scale_, scaler.var_, scaler.n_samples_seen_, rule)
        if np.array_equal(numeric.transform(probe), expected):
            return rule
    raise ValueError('The source normalization arithmetic is unsupported')


def write_anchor(folder, transform, weight, bias, reference_names):
    folder = Path(folder)
    weight, bias = np.asarray(weight), np.asarray(bias)
    names, genes = list(reference_names), list(transform.genes_)
    if (not names or len(set(names)) != len(names) or transform.fitted_names_ != names
            or len(set(genes)) != len(genes) or len(genes) != transform.n_genes
            or weight.shape != (31, len(genes)) or bias.shape != (31,)
            or weight.dtype != np.float32 or bias.dtype != np.float32
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError('Frozen FP32 anchor and fitted feature identities must agree')
    arrays = {'weight': weight, 'bias': bias, 'genes': np.asarray(genes, dtype=str)}
    arithmetic = {}
    for prefix, scaler in [('RNA', transform.scaler_), ('QC', transform.qc_scaler_)]:
        if getattr(scaler, 'with_mean', True) is not True or getattr(scaler, 'with_std', True) is not True:
            raise ValueError('The registered normalization requires centering and scaling')
        for key in ('mean_', 'scale_', 'var_', 'n_samples_seen_'):
            arrays[prefix + '_' + key.rstrip('_')] = np.asarray(getattr(scaler, key))
        arithmetic[prefix] = scaler_arithmetic(scaler)
    np.savez_compressed(folder / 'anchor.npz', **arrays)
    write_json(folder / 'transform.json', {'schema': 'reference_transform_numeric_v2',
               'n_genes': len(genes), 'reference_names': names,
               'scaler_arithmetic': arithmetic,
               'normalization': 'selected-gene library10000 log1p; reference StandardScaler',
               'independently_fitted': True, 'residual_training_seen': False})


def load_anchor(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'transform.json').read_text())
    if (metadata['schema'] != 'reference_transform_numeric_v2' or not metadata['independently_fitted']
            or metadata['residual_training_seen']):
        raise ValueError('An independently fitted reference transform is required')
    with np.load(folder / 'anchor.npz', allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive.files}
    genes, weight, bias = values['genes'], values['weight'], values['bias']
    names = metadata['reference_names']
    if (genes.ndim != 1 or genes.dtype.kind != 'U' or len(set(genes)) != len(genes)
            or len(genes) != metadata['n_genes'] or len(set(names)) != len(names) or not names
            or weight.shape != (31, len(genes)) or bias.shape != (31,)
            or weight.dtype != np.float32 or bias.dtype != np.float32
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError('Portable anchor axes, values or names differ')
    transform = ReferenceTransform(len(genes))
    transform.genes_, transform.fitted_names_ = tuple(genes.tolist()), names
    scalers = []
    for prefix in ('RNA', 'QC'):
        scalers.append(FrozenScaler(*(values[prefix + '_' + k] for k in ('mean', 'scale', 'var', 'n_samples_seen')),
                                    arithmetic=metadata['scaler_arithmetic'][prefix]))
    if scalers[0].n_features_in_ != len(genes) or scalers[1].n_features_in_ != 4:
        raise ValueError('Portable feature/QC dimensions differ')
    transform.scaler_, transform.qc_scaler_ = scalers
    return {'model': SimpleNamespace(coef_=weight, intercept_=bias), 'transform': transform,
            'reference_names': names, 'independently_fitted': True, 'residual_training_seen': False}
