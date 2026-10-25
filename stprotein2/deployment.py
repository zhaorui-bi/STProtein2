"""Convert independently verified research models into small standalone bundles."""
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data import PANEL, ReferenceTransform
from .io import freeze_run, sha256, verify_marker, write_json
from .portable import write_anchor


def read(path):
    return json.loads(Path(path).read_text())


class _DiscardedRouter:
    """Export-only placeholder; never part of a deployed bundle."""


def _read_legacy_preprocessing(path):
    from joblib.numpy_pickle import NumpyUnpickler

    class ConversionReader(NumpyUnpickler):
        def find_class(self, module, name):
            if module == 'stprotein2.data' and name == 'ReferenceTransform':
                return ReferenceTransform
            if module == 'stprotein2.graph' and name == 'FrozenRouter':
                return _DiscardedRouter
            if module.startswith('stprotein2'):
                raise ValueError('Unsupported historical preprocessing class: ' + module + '.' + name)
            return super().find_class(module, name)

    with Path(path).open('rb') as stream:
        if stream.read(1) != b'\x80':
            raise ValueError('Historical preprocessing must be an uncompressed joblib artifact')
        stream.seek(0)
        kwargs = {'mmap_mode': None}
        if 'ensure_native_byte_order' in inspect.signature(NumpyUnpickler).parameters:
            kwargs['ensure_native_byte_order'] = True
        result = ConversionReader(str(path), stream, **kwargs).load()
    if (not isinstance(result, dict) or set(result) != {'transform', 'router', 'train_config'}
            or not isinstance(result['transform'], ReferenceTransform)
            or not isinstance(result['router'], _DiscardedRouter)):
        raise ValueError('Historical preprocessing does not match the registered ordinary model')
    return result


def export_research(run, verification, output):
    """Read frozen weights/metadata only; never read reference/query expression files."""
    import torch
    from .neural import CompactResidual

    run, verification, output = [Path(p).resolve() for p in (run, verification, output)]
    if output.exists():
        raise FileExistsError('Choose a fresh bundle directory')
    if output == run or output in run.parents or run in output.parents:
        raise ValueError('Bundle and research run must be separate directories')
    attestation = read(verification / 'verified_complete.json')
    report_path = verification / 'report.json'
    if attestation['status'] != 'passed' or sha256(report_path) != attestation['report_sha256']:
        raise ValueError('A completed independent verification report is required')
    report = read(report_path)
    if (Path(report['result_folder']).resolve() != run or report['protein_reads_during_replay']
            or report['query_protein_used_for_selection'] is not False
            or report.get('numerical_evaluation_only') is not True
            or report['script_sha256'] != attestation['script_sha256']
            or report['verification_protocol_sha256'] != attestation['verification_protocol_sha256']):
        raise ValueError('Independent verification identity or RNA-only boundary differs')
    marker = read(run / 'prediction_complete.json')
    if marker['query_protein_read_for_prediction'] is not False:
        raise ValueError('The source prediction was not RNA-only')
    required = ('manifest.json', 'model.pt', 'preprocessing.joblib', 'gate_selection.json')
    identities = {}
    for name in required:
        path = run / name
        expected = marker['deployment_artifacts'][name]
        if sha256(path) != expected or report['hashes'].get(str(path)) != expected:
            raise ValueError('Verified deployment artifact changed: ' + name)
        identities[name] = expected
    if identities['manifest.json'] != marker['manifest_sha256']:
        raise ValueError('Source manifest completion differs')
    protocol = read(run / 'manifest.json')['protocol']
    if protocol['query_labels_for_selection'] is not False:
        raise ValueError('Source used query labels for selection')
    mapping, names, queries = protocol['groups'], protocol['refit_names'], protocol['query_names']
    query_groups = {mapping[n] for n in queries}
    if (not queries or len(query_groups) != 1 or set(queries) != {n for n in mapping if mapping[n] in query_groups}
            or set(names) != set(mapping) - set(queries) or len(set(names)) != len(names)):
        raise ValueError('Complete source holdout is required for development export')
    prep = _read_legacy_preprocessing(run / 'preprocessing.joblib')
    cfg = prep['train_config']
    if (cfg['execution_mode'] != 'full' or cfg['graph_k'] != 6 or cfg['graph_radius'] != 2.5
            or cfg['spatial_alpha'] != .5 or prep['transform'].fitted_names_ != names):
        raise ValueError('Preprocessing or physical basis differs from the compact contract')
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    residual, conversion = CompactResidual.from_research_checkpoint(saved)
    if (float(saved['state_dict']['spatial_alpha']) != .5 or saved['spatial_alpha'] != .5
            or residual.cfg.n_genes != prep['transform'].n_genes):
        raise ValueError('Frozen model basis or feature dimensions differ')
    gates = read(run / 'gate_selection.json')
    if tuple(gates['proteins']) != PANEL:
        raise ValueError('Reference gate protein panel differs')
    for key in ('gamma_state', 'gamma_local'):
        np.testing.assert_array_equal(getattr(residual, key).numpy(), gates[key])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stprotein2-export-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'bundle'; stage.mkdir()
        write_anchor(stage, prep['transform'], saved['state_dict']['anchor.weight'].numpy(),
                     saved['state_dict']['anchor.bias'].numpy(), names)
        torch.save({'schema': 'compact_residual_weights_v1', 'config': asdict(residual.cfg),
                    'state_dict': residual.state_dict()}, stage / 'residual.pt')
        write_json(stage / 'conversion.json', {'scope': 'verified development-model export', **conversion,
                   'source_artifact_sha256': identities, 'verification_report_sha256': sha256(report_path),
                   'verification_marker_sha256': sha256(verification / 'verified_complete.json'),
                   'historical_router_deployed': False, 'measured_protein_access_during_export': False})
        freeze_run(stage, {'stage': 'compact_reference_residual_bundle_v1', 'reference_names': names,
                   'groups': {n: mapping[n] for n in names}, 'n_genes': residual.cfg.n_genes,
                   'proteins': list(PANEL), 'spatial_alpha': .5, 'graph_k': 6, 'graph_radius': 2.5,
                   'anchor_arithmetic': 'FP64_multiply_then_FP32', 'neural_head': residual.cfg.residual_head,
                   'residual_config': asdict(residual.cfg), 'query_ADT_read_or_hashed': False,
                   'training_setting': {k: cfg[k] for k in ('seed', 'lr', 'residual_target', 'teacher_mode')},
                   'source_verification_sha256': sha256(report_path),
                   'final_predictive_architecture_frozen': False})
        write_json(stage / 'bundle_complete.json', {'manifest_sha256': sha256(stage / 'manifest.json'),
                   'artifacts': {n: sha256(stage / n) for n in ('anchor.npz', 'transform.json', 'residual.pt', 'conversion.json')},
                   'query_ADT_read_or_hashed': False})
        stage.rename(output)
    return output


def export_baseline(run, output):
    """Strip a completed clean B0 run to its required runtime artifacts."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output == run or output in run.parents or run in output.parents:
        raise ValueError('Use a fresh separate bundle directory')
    from .pipeline import load_bundle
    load_bundle(run)
    marker = verify_marker(run, 'bundle_complete.json')
    if read(run / 'manifest.json')['protocol']['stage'] != 'clean_spatial_ridge_foundation':
        raise ValueError('A clean B0 training run is required')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stprotein2-export-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'bundle'; stage.mkdir()
        for name in ('manifest.json', 'bundle_complete.json', *marker['artifacts']):
            shutil.copyfile(run / name, stage / name)
        verify_marker(stage, 'bundle_complete.json')
        stage.rename(output)
    return output
