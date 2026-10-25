"""Nested source-held epoch and gate selection, followed by compact refitting."""
from dataclasses import asdict, replace
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .anchor import fit_anchor
from .data import Cohort, PANEL
from .gates import component_losses, select_gates
from .graph import physical_graph
from .io import freeze_run, sha256, verify_marker, write_json
from .neural import ResidualConfig
from .pipeline import check_files
from .portable import write_anchor
from .provenance import snapshot_sources
from .training import TrainConfig, TrainingStudent, execution_record, fit, prediction_components, save_torch


def read(path):
    return json.loads(Path(path).read_text())


def write_bound(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Completed selection record changed: ' + str(path))
    else:
        write_json(path, value)


def anchor_record(cohort, names, anchor, folder):
    groups = sorted({cohort.groups[n] for n in names})
    rows = []
    files = sorted(folder.glob('inner_*.json'))
    if len(files) != len(groups) or anchor['selection']['groups'] != groups:
        raise ValueError('Anchor CV coverage differs')
    for group, path in zip(groups, files, strict=True):
        row = read(path)
        training = [n for n in names if cohort.groups[n] != group]
        validation = [n for n in names if cohort.groups[n] == group]
        if (row['training'] != training or row['validation'] != validation
                or row['preprocessing_fitted_names'] != training):
            raise ValueError('Anchor CV source exclusion differs')
        rows.append(row['mse'])
    np.testing.assert_array_equal(rows, anchor['selection']['group_mse'])
    return {'reference_names': names, 'selection': anchor['selection'],
            'files_sha256': {str(p.resolve()): sha256(p) for p in
                             [folder / 'anchor.joblib', folder / 'manifest.json', folder / 'complete.json', *files]}}


def fit_block(cohort, names, held, model_config, config, output, cache, device):
    if (set(names) & set(held) or {cohort.groups[n] for n in names} & {cohort.groups[n] for n in held}):
        raise ValueError('Inner validation sources overlap training')
    anchor, folder = fit_anchor(cohort, names, model_config.n_genes, cache, config.threads)
    record = anchor_record(cohort, names, anchor, folder)
    references = cohort.references(names, anchor['transform'])
    validation = cohort.references(held, anchor['transform'])
    model, report = fit(references, validation, anchor, model_config, config, output, device)
    write_bound(Path(output) / 'anchor_record.json', record)
    return model, report, anchor


def select_and_refit(cohort, names, model_config, config, output, cache, device):
    groups = sorted({cohort.groups[n] for n in names})
    if len(groups) < 3:
        raise ValueError('Nested compact fitting requires three allowed reference sources')
    output = Path(output)
    selected = []
    for i, group in enumerate(groups):
        held = [n for n in names if cohort.groups[n] == group]
        allowed = [n for n in names if cohort.groups[n] != group]
        model, report, _ = fit_block(cohort, allowed, held, model_config, config,
                                     output / f'inner_{i:02d}', cache, device)
        selected.append({'group': group, 'epoch': report['selected_epoch'],
                         'validation_mse': report['reference_validation_mse']})
        del model; gc.collect()
    epochs = int(np.median([r['epoch'] for r in selected]))
    write_bound(output / 'epoch_selection.json', {'inner_groups': selected, 'selected_epoch': epochs,
                'rule': 'integer floor of median selected epochs'})
    model, report, anchor = fit_block(cohort, names, [], model_config, replace(config, epochs=epochs),
                                      output / 'refit', cache, device)
    return model, report, anchor


def gate_evidence(cohort, names, model_config, config, output, cache, device):
    groups = sorted({cohort.groups[n] for n in names})
    output = Path(output)
    evidence = {}
    if len(groups) >= 4:
        for i, group in enumerate(groups):
            allowed = [n for n in names if cohort.groups[n] != group]
            held = [n for n in names if cohort.groups[n] == group]
            fold = output / f'gate_oof_{i:02d}'
            marker = fold / 'gate_complete.json'
            if marker.exists():
                saved = read(fold / 'component_losses.json')
                if (sha256(fold / 'component_losses.json') != read(marker)['component_losses_sha256']
                        or saved['held_group'] != group or saved['allowed_names'] != allowed or saved['held_names'] != held
                        or set(saved['prediction_sha256']) != set(held)
                        or saved['held_labels_used_for_epoch_selection'] is not False):
                    raise ValueError('Cached gate source exclusion or component evidence differs')
                check_files(fold, {f'{n}_prediction.npz': h for n, h in saved['prediction_sha256'].items()})
                for fit_marker in fold.rglob('fit_complete.json'):
                    verify_marker(fit_marker.parent, fit_marker.name)
                evidence[group] = np.asarray(saved['group_mean_losses'])
                continue
            model, _, anchor = select_and_refit(cohort, allowed, model_config, config, fold, cache, device)
            losses, hashes = [], {}
            for name in held:
                before = list(cohort.protein_reads)
                view = cohort.input(name, anchor['transform'])
                arrays = prediction_components(model, anchor, view, physical_graph(view.coordinates), device)
                if before != cohort.protein_reads:
                    raise RuntimeError('Gate OOF prediction read held protein')
                destination = fold / f'{name}_prediction.npz'
                np.savez_compressed(destination, **arrays, proteins=np.asarray(PANEL), barcodes=view.barcodes)
                hashes[name] = sha256(destination)
                # Held labels enter only after their RNA-only prediction freezes.
                losses.append(component_losses(cohort.measured_protein(name), arrays))
            evidence[group] = np.mean(losses, axis=0)
            write_bound(fold / 'component_losses.json', {'held_group': group, 'allowed_names': allowed,
                        'held_names': held, 'group_mean_losses': evidence[group].tolist(),
                        'prediction_sha256': hashes, 'held_labels_used_for_epoch_selection': False})
            write_bound(marker, {'component_losses_sha256': sha256(fold / 'component_losses.json')})
            del model; gc.collect()
    result = select_gates(evidence, PANEL)
    result['available_reference_groups'] = groups
    result['status'] = ('selected on predictions excluding each held group from training and epoch selection'
                        if len(groups) >= 4 else 'closed: fewer than four reference groups; no gate OOF fits attempted')
    write_bound(output / 'gate_selection.json', result)
    return result


def audit_training(output, names, groups):
    output = Path(output)
    fit_markers = sorted(output.rglob('fit_complete.json'))
    n = len({groups[name] for name in names})
    expected = (n * n if n >= 4 else 0) + n + 1
    if len(fit_markers) != expected:
        raise ValueError('Nested fit coverage differs from the declared source count')
    files = {}
    for marker in fit_markers:
        verify_marker(marker.parent, marker.name)
        protocol = read(marker.parent / 'manifest.json')['protocol']
        fitted, held = protocol['reference_identity'], protocol['validation_identity']
        used = fitted + held
        if (not {v['name'] for v in used} <= set(names)
                or any(v['group'] != groups[v['name']] for v in used)
                or {v['group'] for v in fitted} & {v['group'] for v in held}):
            raise ValueError('An excluded source entered a nested fit')
        relative = marker.relative_to(output)
        parts = list(relative.parts)
        allowed = list(names)
        if parts[0].startswith('gate_oof_'):
            group = sorted({groups[name] for name in names})[int(parts.pop(0).removeprefix('gate_oof_'))]
            allowed = [name for name in names if groups[name] != group]
        if parts == ['refit', 'fit_complete.json']:
            expected_fit, expected_held = allowed, []
        elif len(parts) == 2 and parts[0].startswith('inner_') and parts[1] == 'fit_complete.json':
            group = sorted({groups[name] for name in allowed})[int(parts[0].removeprefix('inner_'))]
            expected_fit = [name for name in allowed if groups[name] != group]
            expected_held = [name for name in allowed if groups[name] == group]
        else:
            raise ValueError('Unexpected nested fit location')
        if [v['name'] for v in fitted] != expected_fit or [v['name'] for v in held] != expected_held:
            raise ValueError('Gate or epoch holdout did not exclude its complete source')
        files[str(relative)] = sha256(marker)
    for p in output.rglob('*'):
        if (p.is_file() and (p.name in {'anchor_record.json', 'epoch_selection.json', 'component_losses.json', 'gate_complete.json', 'gate_selection.json'}
                            or ('gate_oof_' in str(p.relative_to(output)) and p.name.endswith('_prediction.npz'))
                            or ('anchor_cache' in p.parts and p.name in {'anchor.joblib', 'complete.json', 'manifest.json'})
                            or ('anchor_cache' in p.parts and p.name.startswith('inner_') and p.suffix == '.json'))):
            files[str(p.relative_to(output))] = sha256(p)
    return {'fit_count': len(fit_markers), 'fit_markers': [str(p.relative_to(output)) for p in fit_markers],
            'files_sha256': files, 'query_ADT_read_or_hashed': False}


def train(registry, groups, output, *, query=None, model_config=ResidualConfig(), config=TrainConfig(), device='cpu'):
    """Train a standalone compact bundle using only declared allowed references."""
    mapping = read(groups)['groups']
    cohort = Cohort(registry, mapping)
    if query is not None and query not in mapping:
        raise ValueError('Unknown development query sample')
    names = [n for n in mapping if query is None or mapping[n] != mapping[query]]
    if len({mapping[n] for n in names}) < 3:
        raise ValueError('Nested compact fitting requires three allowed reference sources')
    torch.set_num_threads(config.threads)
    execution = execution_record(device)
    output = Path(output).resolve()
    if output.exists() and not (output / 'manifest.json').exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh directory or a matching resumable training run')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.train.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_run(output, {'stage': 'compact_reference_residual_bundle_v1', 'reference_names': names,
                   'groups': {n: mapping[n] for n in names}, 'n_genes': model_config.n_genes,
                   'proteins': list(PANEL), 'spatial_alpha': .5, 'graph_k': 6, 'graph_radius': 2.5,
                   'anchor_arithmetic': 'FP64_multiply_then_FP32', 'neural_head': model_config.residual_head,
                   'residual_config': asdict(model_config), 'training_setting': asdict(config),
                   'origin': 'clean_nested_reference_training', 'initialization': TrainingStudent.initialization,
                   'execution': execution,
                   'input_sha256': {cohort.specs[n][m]: sha256(cohort.specs[n][m]) for n in names for m in ('rna', 'adt')},
                   'gate_rule': 'reference OOF: 4 groups / 1% relative MSE / 60% groups improved',
                   'query_ADT_read_or_hashed': False, 'query_names': [n for n in mapping if n not in names],
                   'final_predictive_architecture_frozen': False})
        if (output / 'bundle_complete.json').exists():
            marker = verify_marker(output, 'bundle_complete.json')
            if sha256(output / 'training_complete.json') != marker['training_complete_sha256']:
                raise ValueError('Completed training audit changed')
            audit = read(output / 'training_complete.json')
            check_files(output, audit['files_sha256'])
            for relative in audit['fit_markers']:
                path = output / relative; verify_marker(path.parent, path.name)
            return output
        torch.set_num_threads(config.threads)
        cache = output / 'anchor_cache'
        with threadpool_limits(config.threads):
            gates = gate_evidence(cohort, names, model_config, config, output, cache, device)
            model, report, anchor = select_and_refit(cohort, names, model_config, config, output, cache, device)
        if set(cohort.protein_reads) - set(names):
            raise RuntimeError('An excluded query protein entered training or selection')
        for field in ('gamma_state', 'gamma_local'):
            getattr(model, field).copy_(torch.as_tensor(gates[field], device=device, dtype=torch.float32))
        write_anchor(output, anchor['transform'], np.asarray(anchor['model'].coef_, dtype=np.float32),
                     np.asarray(anchor['model'].intercept_, dtype=np.float32), names)
        save_torch(output / 'residual.pt', {'schema': 'compact_residual_weights_v1', 'config': asdict(model_config),
                   'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}})
        write_json(output / 'conversion.json', {'scope': 'clean nested training; auxiliary head removed',
                   'historical_router_deployed': False, 'auxiliary_direct_head_deployed': False,
                   'selected_epoch': report['selected_epoch'], 'measured_query_protein_used': False,
                   **model.specification()})
        snapshot_sources(output)
        audit = audit_training(output, names, mapping)
        audit['protein_reads'] = cohort.protein_reads
        write_json(output / 'training_complete.json', audit)
        write_json(output / 'bundle_complete.json', {'manifest_sha256': sha256(output / 'manifest.json'),
                   'artifacts': {n: sha256(output / n) for n in ('anchor.npz', 'transform.json', 'residual.pt', 'conversion.json')},
                   'training_complete_sha256': sha256(output / 'training_complete.json'),
                   'query_ADT_read_or_hashed': False})
    return output
