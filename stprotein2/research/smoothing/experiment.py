"""Independent nested anchor lineage, reference selection, RNA-only inference."""
import json
from pathlib import Path

import joblib
import numpy as np

from stprotein2.data import PANEL
from stprotein2.graph import physical_graph, digest_array
from stprotein2.io import freeze_run, sha256, write_json
from stprotein2.training.residual_targets import get_anchor, predict_anchor
from stprotein2.models.smoothing import ALPHAS, MODES, spatial_residual, apply, permute_graph, select_alpha


def predict(bundle, view, alpha, mode='physical', seed=2025):
    if mode not in MODES:
        raise ValueError('Unknown spatial control')
    base = predict_anchor(bundle, view)
    graph = physical_graph(view.coordinates, k=6, radius=2.5)
    permutation = None
    if mode == 'permuted':
        graph, permutation = permute_graph(graph, view.barcodes, view.name, seed)
    residual = spatial_residual(base, graph)
    return {'prediction': apply(base, residual, alpha), 'base': base, 'local_delta': residual,
            'barcodes': view.barcodes, 'proteins': np.asarray(PANEL),
            'graph_sha256': graph.fingerprint,
            'permutation_sha256': digest_array(permutation) if permutation is not None else None,
            'isolated_nodes': int(np.sum(graph.degree == 0))}


def anchor_lineage(cohort, names, genes):
    bundle, path = get_anchor(cohort, names, genes)
    if bundle['reference_names'] != names or bundle['transform'].fitted_names_ != names:
        raise ValueError('Anchor reference identity differs')
    folds = []
    for source in sorted(path.glob('inner_*.json')):
        row = json.loads(source.read_text())
        if (set(row['training'] + row['validation']) != set(names)
                or row['preprocessing_fitted_names'] != row['training']
                or {cohort.groups[n] for n in row['training']} & {cohort.groups[n] for n in row['validation']}):
            raise ValueError('Anchor internal CV lineage is inconsistent')
        folds.append({'file': source.name, 'sha256': sha256(source),
                      'training': row['training'], 'validation': row['validation']})
    if len(folds) != len({cohort.groups[n] for n in names}):
        raise ValueError('Incomplete anchor CV lineage')
    return bundle, {'anchor_sha256': sha256(path / 'anchor.joblib'), 'reference_names': names,
                    'reference_groups': sorted({cohort.groups[n] for n in names}), 'CV': folds}


def run_query(cohort, query, output, *, genes=4000, seed=2025):
    if getattr(cohort, 'anchor_cache_policy', None) != 'reuse_frozen':
        raise ValueError('This control requires independent completed anchors; it never silently refits them')
    groups = cohort.groups
    queries = [n for n in groups if groups[n] == groups[query]]
    names = [n for n in groups if groups[n] != groups[query]]
    references = sorted({groups[n] for n in names})
    if len(references) < 3:
        raise ValueError('At least three reference sources needed for nested independent anchors')
    output = Path(output)
    # Each graph variant is a separate scored run with the same anchor protocol.
    from stprotein2.provenance import sources
    dependencies = sources()
    fixed = {'stage': 'seen_development_spatial_shrinkage', 'groups': groups, 'query_names': queries,
             'refit_names': names, 'n_genes': genes, 'seed': seed, 'alpha_grid': ALPHAS,
             'selection': 'reference-only held-source equal-group/equal-slice MSE; one alpha shared by all proteins',
             'graph': 'symmetric k6 radius2.5; one-hop neighbor-mean minus self; mean-preserving across nonisolated slice nodes',
             'permuted_control': 'barcode-stable graph node relabeling; same topology and degree multiset; alpha reselected',
             'neural_per_protein_gate_rule_changed': False, 'query_labels_for_selection': False,
             'input_sha256': {s[m]: sha256(s[m]) for s in cohort.specs.values() for m in ('rna', 'adt')},
             'dependency_sources': dependencies}
    completed = {}
    for mode in MODES:
        folder = output / mode
        freeze_run(folder, {**fixed, 'mode': mode})
        marker = folder / 'prediction_complete.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if any(sha256(folder / p) != h for p, h in record['artifacts_sha256'].items()):
                raise RuntimeError('Completed spatial control artifact changed')
            completed[mode] = record
    if len(completed) == len(MODES):
        return completed
    records = {m: [] for m in MODES}
    provenance = []
    before = len(cohort.protein_reads)
    for held in references:
        training = [n for n in names if groups[n] != held]
        validation = [n for n in names if groups[n] == held]
        bundle, lineage = anchor_lineage(cohort, training, genes)
        if held in lineage['reference_groups'] or groups[query] in lineage['reference_groups']:
            raise RuntimeError('Held or query source entered an OOF anchor')
        for name in validation:
            view = cohort.input(name, bundle['transform'])
            outputs = {mode: predict(bundle, view, 0., mode, seed) for mode in MODES}
            target = cohort.measured_protein(name)
            for mode, row in outputs.items():
                records[mode].append({'name': name, 'group': held, 'base': row['base'],
                                      'residual': row['local_delta'], 'target': target})
            provenance.append({'held_source': held, 'sample': name, 'anchor': lineage,
                               'input_sha256': view.fingerprint(), 'barcodes_sha256': digest_array(view.barcodes),
                               'graphs': {m: {'graph_sha256': v['graph_sha256'], 'permutation_sha256': v['permutation_sha256']}
                                          for m, v in outputs.items()}})
    bundle, full_lineage = anchor_lineage(cohort, names, genes)
    if set(cohort.protein_reads[before:]) & set(queries):
        raise RuntimeError('Query protein accessed during spatial control selection')
    for mode in MODES:
        if mode in completed:
            continue
        folder = output / mode
        selected = select_alpha(records[mode])
        selected.update(OOF_provenance=provenance, deployment_anchor=full_lineage)
        selection = folder / 'selection.json'
        # Selection is deterministic; preserve a previously saved matching file.
        if selection.exists() and json.loads(selection.read_text()) != selected:
            raise RuntimeError('Spatial reference selection changed during recovery')
        write_json(selection, selected)
        exported = {'anchor': bundle, 'transform': bundle['transform'], 'alpha': selected['alpha'],
                    'mode': mode, 'seed': seed, 'reference_names': names, 'proteins': PANEL}
        prep = folder / 'preprocessing.joblib'
        if prep.exists():
            if joblib.hash(joblib.load(prep)) != joblib.hash(exported):
                raise RuntimeError('Spatial deployment bundle changed')
        else:
            joblib.dump(exported, prep)
        reads = list(cohort.protein_reads); hashes = {}; graph_records = {}
        for name in queries:
            view = cohort.input(name, bundle['transform'])
            row = predict(bundle, view, selected['alpha'], mode, seed)
            path = folder / 'predictions' / name / 'prediction.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: row[k] for k in ('prediction', 'base', 'local_delta', 'barcodes', 'proteins')})
            hashes[name] = sha256(path)
            graph_records[name] = {k: row[k] for k in ('graph_sha256', 'permutation_sha256', 'isolated_nodes')}
        if reads != cohort.protein_reads:
            raise RuntimeError('Spatial prediction read query protein')
        write_json(folder / 'graphs.json', graph_records)
        artifacts = [selection, prep, folder / 'graphs.json', *sorted(folder.glob('predictions/*/prediction.npz'))]
        record = {'queries': queries, 'sha256': hashes, 'query_protein_read_for_prediction': False,
                  'protein_reads_before_prediction': cohort.protein_reads[before:],
                  'artifacts_sha256': {str(p.relative_to(folder)): sha256(p) for p in artifacts}}
        write_json(folder / 'prediction_complete.json', record)
        completed[mode] = record
    return completed
