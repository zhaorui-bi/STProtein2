"""Leave-source-group-out development with separately nested OOF gate fitting.

This is intentionally more expensive than choosing gates on the same validation
labels used for early stopping. All seven README slices are already seen data.
"""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .config import ModelConfig, TrainConfig, from_dict
from .data import Cohort, PANEL, ReferenceTransform
from .gates import component_losses, select_gates
from .graph import FrozenRouter
from .io import freeze_run, save_torch, sha256, write_json
from .model import STProtein2
from .trainer import fit, predict


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = STProtein2(from_dict(ModelConfig, saved['model_config'])).to(device)
    model.load_state_dict(saved['state_dict']); model.eval()
    return model


def fit_block(cohort, names, validation_names, model_cfg, train_cfg, output, device):
    """A fresh preprocessing/router is fitted per training subset, persisted on disk."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'training.json').exists():
        report = json.loads((output/'training.json').read_text())
        if report['reference_names'] != names or report['inner_validation'] != validation_names:
            raise ValueError('Completed fit identity differs')
        prep = joblib.load(output/'preprocessing.joblib')
        return load_model(output/'model.pt', device), report, prep['transform'], prep['router']
    transform = ReferenceTransform(model_cfg.n_genes).fit([cohort.rna(n) for n in names], names)
    references = cohort.references(names, transform, train_cfg.count_keep_rates, train_cfg.count_views_per_rate, train_cfg.seed)
    router = FrozenRouter(seed=train_cfg.seed).fit([s.rna.x for s in references], [s.rna.group for s in references])
    validation = cohort.references(validation_names, transform)
    joblib.dump({'transform': transform, 'router': router, 'reference_names': names,
                 'train_config': train_cfg.to_dict()}, output/'preprocessing.joblib')
    model, report = fit(references, model_cfg, train_cfg, router, output, device, validation=validation)
    return model, report, transform, router


def select_and_refit(cohort, names, model_cfg, train_cfg, output, device):
    """Select a median epoch across inner groups, then refit all allowed names."""
    groups = sorted({cohort.groups[n] for n in names})
    if len(groups) < 2:
        raise ValueError('Nested epoch selection requires at least two reference groups')
    output = Path(output); selected = []
    for i, group in enumerate(groups):
        validation = [n for n in names if cohort.groups[n] == group]
        training = [n for n in names if cohort.groups[n] != group]
        print(f'Inner group {i+1}/{len(groups)}: validation={validation}', flush=True)
        model, report, _, _ = fit_block(cohort, training, validation, model_cfg, train_cfg,
                                        output/f'inner_{i:02d}', device)
        selected.append({'group': group, 'epoch': report['selected_epoch'],
                         'validation_mse': report['reference_validation_mse']})
        del model; gc.collect()
    epochs = max(1, int(np.median([r['epoch'] for r in selected])))
    chosen_train = replace(train_cfg, epochs=epochs, warmup_epochs=min(train_cfg.warmup_epochs, epochs))
    write_json(output/'epoch_selection.json', {'inner_groups': selected, 'selected_epoch': epochs,
                                              'rule': 'integer floor of median selected epochs'})
    print(f'Refit {len(names)} references, {len(groups)} groups, {epochs} epochs', flush=True)
    model, report, transform, router = fit_block(cohort, names, [], model_cfg, chosen_train,
                                               output/'refit', device)
    return model, report, transform, router, chosen_train


def gate_evidence(cohort, names, model_cfg, train_cfg, output, device):
    groups = sorted({cohort.groups[n] for n in names})
    output = Path(output)
    if len(groups) < 4:
        result = select_gates({}, PANEL)
        result['available_reference_groups'] = groups
        result['status'] = 'closed: fewer than four reference groups; no gate OOF fits attempted'
        write_json(output/'gate_selection.json', result)
        return result
    evidence = {}
    for i, group in enumerate(groups):
        allowed = [n for n in names if cohort.groups[n] != group]
        held = [n for n in names if cohort.groups[n] == group]
        fold = output/f'gate_oof_{i:02d}'
        if (fold/'component_losses.json').exists():
            saved = json.loads((fold/'component_losses.json').read_text())
            if saved['held_group'] != group or saved['allowed_names'] != allowed:
                raise ValueError('Gate evidence identity differs')
            evidence[group] = np.asarray(saved['group_mean_losses'])
            continue
        # Epoch selection here only sees allowed references. Held labels are
        # opened after prediction and never used by the fit or its early stop.
        model, _, transform, router, selected_train = select_and_refit(cohort, allowed, model_cfg, train_cfg, fold, device)
        losses, prediction_hashes = [], {}
        for name in held:
            before = list(cohort.protein_reads)
            arrays, _ = predict(model, cohort.input(name, transform), router, selected_train, device)
            if before != cohort.protein_reads:
                raise RuntimeError('Gate OOF prediction read held protein')
            path = fold/f'{name}_prediction.npz'
            np.savez_compressed(path, **arrays, proteins=np.asarray(PANEL))
            prediction_hashes[name] = sha256(path)
            losses.append(component_losses(cohort.measured_protein(name), arrays))
        evidence[group] = np.mean(losses, axis=0)
        write_json(fold/'component_losses.json', {'held_group': group, 'allowed_names': allowed,
                    'held_names': held, 'group_mean_losses': evidence[group].tolist(),
                    'prediction_sha256': prediction_hashes, 'held_labels_used_for_epoch_selection': False})
        del model; gc.collect()
    result = select_gates(evidence, PANEL)
    result['available_reference_groups'] = groups
    result['status'] = 'selected on predictions excluding each held group from training and epoch selection'
    write_json(output/'gate_selection.json', result)
    return result


def run_nested(args):
    spec = json.loads(Path(args.config).read_text())
    groups = json.loads(Path(args.groups).read_text())['groups']
    cohort = Cohort(args.registry, groups)
    if any(accession in str(entry) for entry in cohort.specs.values() for accession in ('GSE278662', 'GSE322883')):
        raise ValueError('Reserved cohorts cannot enter the development runner')
    model_cfg = from_dict(ModelConfig, spec.get('model', {}))
    train_cfg = from_dict(TrainConfig, spec.get('train', {}))
    if args.query not in groups:
        raise ValueError('Unknown outer query')
    query_group = groups[args.query]
    queries = [n for n in groups if groups[n] == query_group]
    references = [n for n in groups if groups[n] != query_group]
    output = Path(args.output).resolve()
    inputs = {entry[m]: sha256(entry[m]) for entry in cohort.specs.values() for m in ('rna', 'adt')}
    freeze_run(output, {'stage': 'seen_development_nested_leave_source_group_out',
               'model': model_cfg.to_dict(), 'train': train_cfg.to_dict(), 'groups': groups,
               'query_names': queries, 'refit_names': references, 'input_sha256': inputs,
               'query_labels_for_selection': False,
               'gate_protocol': 'reference OOF, held group excluded also from early stopping; 4 groups / 1% / 60%',
               'outer_score_scope': 'development only; repeated inspection is not independent confirmation'})
    start = time.monotonic()
    with threadpool_limits(train_cfg.threads):
        gates = gate_evidence(cohort, references, model_cfg, train_cfg, output, args.device)
        model, _, transform, router, chosen_train = select_and_refit(cohort, references, model_cfg, train_cfg, output, args.device)
        model.set_gates(gates['gamma_state'], gates['gamma_local'])
        save_torch(output/'model.pt', {'model_config': model_cfg.to_dict(), 'state_dict': model.state_dict()})
        joblib.dump({'transform': transform, 'router': router, 'train_config': chosen_train.to_dict()}, output/'preprocessing.joblib')
        before = list(cohort.protein_reads); hashes = {}
        for name in queries:
            arrays, diagnostics = predict(model, cohort.input(name, transform), router, chosen_train, args.device)
            arrays['ungated_prediction'] = arrays['base'] + arrays['state_delta'] + arrays['local_delta']
            destination = output/'predictions'/name; destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination/'prediction.npz', **arrays, proteins=np.asarray(PANEL))
            write_json(destination/'diagnostics.json', diagnostics)
            hashes[name] = sha256(destination/'prediction.npz')
        if before != cohort.protein_reads:
            raise RuntimeError('Outer query inference read measured proteins')
        write_json(output/'prediction_complete.json', {'queries': queries, 'sha256': hashes,
                   'protein_reads_before_prediction': before, 'query_protein_read_for_prediction': False,
                   'elapsed_this_invocation_seconds': time.monotonic()-start})
    print('NESTED_PREDICTIONS_COMPLETE '+str(output), flush=True)
