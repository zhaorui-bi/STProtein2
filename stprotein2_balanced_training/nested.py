"""Strictly nested neural correction above the fixed spatial baseline."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from stprotein2.config import ModelConfig, from_dict
from stprotein2.data import Cohort, PANEL
from stprotein2.gates import component_losses, select_gates
from stprotein2.graph import FrozenRouter
from stprotein2.io import freeze_run, save_torch, sha256, write_json
from stprotein2_extended_baselines.runner import verify_marker
from stprotein2_spatial_anchor.prediction import predict
from stprotein2_spatial_anchor.model import SpatialAnchoredSTProtein2
from .provenance import sources, snapshot_sources
from stprotein2_spatial_anchor.targets import build_targets, get_anchor
from .config import BalancedTrainConfig
from stprotein2_balanced_initialization import fit_router
from stprotein2_execution.memory import ExecutionPolicy, configure_execution
from .trainer import fit
from stprotein2_teacher_ablation_v2.identity import content_identity


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = SpatialAnchoredSTProtein2(from_dict(ModelConfig, saved['model_config']), saved['spatial_alpha']).to(device)
    configure_execution(model, ExecutionPolicy(**saved["execution_policy"]))
    model.load_state_dict(saved['state_dict']); model.eval()
    if float(model.spatial_alpha) != saved['spatial_alpha']:
        raise RuntimeError('Checkpoint fixed spatial coefficient differs')
    return model


def fit_block(cohort, names, validation_names, model_cfg, train_cfg, output, device):
    """A fresh preprocessing/router is fitted per training subset, persisted on disk."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if len({cohort.groups[n] for n in names}) < 3:
        raise ValueError('Matched C0/CF0 fits require at least three allowed source groups')
    if set(names) & set(validation_names) or {cohort.groups[n] for n in names} & {cohort.groups[n] for n in validation_names}:
        raise ValueError('Residual validation source overlaps training')
    anchor, anchor_path = get_anchor(cohort, names, model_cfg.n_genes, train_cfg.threads)
    anchor_record = {'path': str(anchor_path), 'reference_names': names,
                     'sha256': sha256(anchor_path/'anchor.joblib'), 'selection': anchor['selection']}
    transform = anchor['transform']
    baselines = build_targets(cohort, names, model_cfg.n_genes, train_cfg, (anchor, anchor_path))
    baseline_record = {name: value.signature() for name, value in baselines.items()}
    references = cohort.references(names, transform, train_cfg.count_keep_rates, train_cfg.count_views_per_rate, train_cfg.seed)
    if train_cfg.initialization == "hierarchical":
        router, router_sampling = fit_router([s.rna for s in references], reference_names=names,
            components=train_cfg.router_components, clusters=train_cfg.router_clusters,
            per_source=train_cfg.router_per_source, seed=train_cfg.seed)
    else:
        router = FrozenRouter(components=train_cfg.router_components, clusters=train_cfg.router_clusters,
            per_group=train_cfg.router_per_source, seed=train_cfg.seed).fit(
                [s.rna.x for s in references], [s.rna.group for s in references])
        router_sampling = {"variant": "historical_source_pooled", "reference_names": names,
            "reference_groups": sorted({s.rna.group for s in references}), "sample_barcodes_recorded": False}
    validation = cohort.references(validation_names, transform)
    model, report = fit(references, model_cfg, train_cfg, router, output, anchor['model'], baselines,
                         device, validation=validation)
    # Identity is checked by fit before existing provenance may be touched.
    for filename, value in [('anchor.json', anchor_record), ('residual_baselines.json', baseline_record)]:
        path = output/filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError('Completed residual fit provenance changed')
        if not path.exists():
            write_json(path, value)
    prep = {'transform': transform, 'router': router, 'reference_names': names,
            'train_config': train_cfg.to_dict(), 'router_sampling': router_sampling}
    prep_path = output/'preprocessing.joblib'
    if prep_path.exists():
        if content_identity(joblib.load(prep_path)) != content_identity(prep):
            raise RuntimeError('Residual fit preprocessing changed')
    else:
        joblib.dump(prep, prep_path)
    prep_marker = output / 'preprocessing_complete.json'
    expected = {'content_identity': content_identity(prep), 'artifact_sha256': sha256(prep_path),
                'reference_names': names, 'identity_format': 'canonical_numpy_content_v1'}
    if prep_marker.exists() and json.loads(prep_marker.read_text()) != expected:
        raise RuntimeError('Stored preprocessing artifact identity changed')
    if not prep_marker.exists():
        write_json(prep_marker, expected)
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
        if (fold/'gate_complete.json').exists():
            marker = json.loads((fold/'gate_complete.json').read_text())
            if sha256(fold/'component_losses.json') != marker['component_losses_sha256']:
                raise RuntimeError('Cached residual gate evidence changed')
            saved = json.loads((fold/'component_losses.json').read_text())
            if saved['held_group'] != group or saved['allowed_names'] != allowed or saved['held_names'] != held:
                raise ValueError('Gate evidence identity differs')
            if set(saved['prediction_sha256']) != set(held) or any(
                    sha256(fold/f'{name}_prediction.npz') != value for name, value in saved['prediction_sha256'].items()):
                raise RuntimeError('Cached residual gate predictions changed')
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
        write_json(fold/'gate_complete.json', {'component_losses_sha256': sha256(fold/'component_losses.json')})
        del model; gc.collect()
    result = select_gates(evidence, PANEL)
    result['available_reference_groups'] = groups
    result['status'] = 'selected on predictions excluding each held group from training and epoch selection'
    write_json(output/'gate_selection.json', result)
    return result


def run_nested(args, prepare_only=False):
    spec = json.loads(Path(args.config).read_text())
    groups = json.loads(Path(args.groups).read_text())['groups']
    # Check feasibility before any data matrix or label access. C0 follows the
    # same cohort requirement to remain a matched control of the CF0 pipeline.
    if len(set(groups.values())) < 6:
        raise ValueError('Strict nested C0/CF0 requires at least six conservative source groups')
    cohort = Cohort(args.registry, groups)
    cohort.anchor_cache = Path(args.anchor_cache).resolve()
    cohort.reuse_anchor_caches = [str(Path(p).resolve()) for p in getattr(args, 'reuse_anchor_cache', ())]
    cohort.anchor_cache_policy = spec.get('anchor_cache_policy', 'fit')
    if cohort.anchor_cache_policy not in {'fit', 'reuse_frozen'}:
        raise ValueError('Unknown anchor cache policy')
    if any(accession in str(entry) for entry in cohort.specs.values() for accession in ('GSE278662', 'GSE322883')):
        raise ValueError('Reserved cohorts cannot enter the development runner')
    model_cfg = from_dict(ModelConfig, spec.get('model', {}))
    train_cfg = from_dict(BalancedTrainConfig, spec.get('train', {}))
    if args.query not in groups:
        raise ValueError('Unknown outer query')
    query_group = groups[args.query]
    queries = [n for n in groups if groups[n] == query_group]
    references = [n for n in groups if groups[n] != query_group]
    output = Path(args.output).resolve()
    inputs = {cohort.specs[n][m]: sha256(cohort.specs[n][m]) for n in references + queries
              for m in (('rna', 'adt') if n in references else ('rna',))}
    freeze_run(output, {'stage': 'seen_development_nested_balanced_initialization',
               'method': train_cfg.variant, 'dependency_source_sha256': sources(),
               'anchor_protocol': 'independent nested reference Ridge; frozen throughout residual training',
               'spatial_basis': {'alpha': train_cfg.spatial_alpha, 'center': 'whole_slice_nonisolated',
                                 'graph_k': train_cfg.graph_k, 'graph_radius': train_cfg.graph_radius},
               'anchor_cache_policy': cohort.anchor_cache_policy,
               'anchor_cache': str(cohort.anchor_cache), 'reuse_anchor_caches': cohort.reuse_anchor_caches,
               'residual_protocol': train_cfg.residual_target,
               'residual_baseline_excluded_from_all_anchor_fitting': train_cfg.residual_target == 'cross_fitted',
               'model': model_cfg.to_dict(), 'train': train_cfg.to_dict(), 'groups': groups,
               'query_names': queries, 'refit_names': references, 'inputs': inputs,
               'query_labels_for_selection': False,
               'gate_protocol': 'reference OOF, held group excluded also from early stopping; 4 groups / 1% / 60%',
               'outer_score_scope': 'development only; repeated inspection is not independent confirmation'})
    snapshot_sources(output)
    completed = output/'prediction_complete.json'
    if completed.exists() and not prepare_only:
        marker = verify_marker(output, 'prediction_complete.json')
        if marker['queries'] != queries or any(sha256(output/'predictions'/name/'prediction.npz') != value
                                              for name, value in marker['sha256'].items()):
            raise RuntimeError('Completed residual predictions changed')
        if set(marker['sha256']) != set(queries):
            raise RuntimeError('Completed residual prediction coverage changed')
        for name, value in marker['deployment_artifacts'].items():
            if sha256(output/name) != value:
                raise RuntimeError('Completed residual deployment artifact changed')
        return
    if prepare_only:
        namesets = [references]
        outer_groups = sorted({groups[n] for n in references})
        namesets.extend([[n for n in references if groups[n] != g] for g in outer_groups])
        if len(outer_groups) >= 4:
            for held in outer_groups:
                allowed = [n for n in references if groups[n] != held]
                namesets.extend([[n for n in allowed if groups[n] != g] for g in outer_groups if g != held])
        if train_cfg.residual_target == 'cross_fitted':
            for training in list(namesets):
                for target_group in sorted({groups[n] for n in training}):
                    namesets.append([n for n in training if groups[n] != target_group])
        seen = set()
        for names in namesets:
            if tuple(names) in seen:
                continue
            seen.add(tuple(names))
            get_anchor(cohort, names, model_cfg.n_genes, train_cfg.threads)
        write_json(output/'anchors_prepared.json', {'allowed_reference_subsets': [list(n) for n in sorted(seen)],
                    'protein_reads': cohort.protein_reads, 'query_protein_read': any(n in queries for n in cohort.protein_reads)})
        if any(n in queries for n in cohort.protein_reads):
            raise RuntimeError('Anchor preparation read outer query protein')
        print('ANCHORS_PREPARED '+str(output), flush=True)
        return
    start = time.monotonic()
    with threadpool_limits(train_cfg.threads):
        gates = gate_evidence(cohort, references, model_cfg, train_cfg, output, args.device)
        model, _, transform, router, chosen_train = select_and_refit(cohort, references, model_cfg, train_cfg, output, args.device)
        model.set_gates(gates['gamma_state'], gates['gamma_local'])
        save_torch(output/'model.pt', {'variant': train_cfg.variant, 'model_config': model_cfg.to_dict(), 'spatial_alpha': train_cfg.spatial_alpha, 'execution_policy': train_cfg.execution_policy.to_dict(), 'state_dict': model.state_dict()})
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
        artifacts = {}
        for marker_path in output.rglob('fit_complete.json'):
            artifacts[str(marker_path.relative_to(output))] = sha256(marker_path)
            for name, checksum in json.loads(marker_path.read_text())['sha256'].items():
                path = marker_path.parent / name
                if sha256(path) != checksum:
                    raise RuntimeError('Completed nested fit changed before prediction freeze')
                artifacts[str(path.relative_to(output))] = checksum
        for prep_marker in output.rglob('preprocessing_complete.json'):
            prep_record = json.loads(prep_marker.read_text())
            prep_file = prep_marker.parent / 'preprocessing.joblib'
            if sha256(prep_file) != prep_record['artifact_sha256']:
                raise RuntimeError('Nested preprocessing artifact changed')
            artifacts[str(prep_marker.relative_to(output))] = sha256(prep_marker)
            artifacts[str(prep_file.relative_to(output))] = prep_record['artifact_sha256']
        for name in ('model.pt', 'preprocessing.joblib', 'gate_selection.json'):
            artifacts[name] = sha256(output / name)
        for name, checksum in hashes.items():
            artifacts[f'predictions/{name}/prediction.npz'] = checksum
        write_json(output/'prediction_complete.json', {'queries': queries, 'sha256': hashes,
                   'manifest_sha256': sha256(output/'manifest.json'), 'artifacts': artifacts,
                   'protein_reads_before_prediction': before, 'query_protein_read_for_prediction': False,
                   'deployment_artifacts': {name: sha256(output/name) for name in
                                            ('model.pt', 'preprocessing.joblib', 'gate_selection.json', 'manifest.json')},
                   'elapsed_this_invocation_seconds': time.monotonic()-start})
    print('SPATIAL_BASIS_RESIDUAL_PREDICTIONS_COMPLETE '+str(output), flush=True)
