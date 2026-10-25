"""Compare complete six-source compact fits with the frozen research runner.

Synthetic data and a small matched budget test software compatibility. These
results are not performance evidence for selecting a biological model.
"""
import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import anndata as ad
import numpy as np
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stprotein2.data import PANEL
from stprotein2.graph import physical_graph, digest_array
from stprotein2.io import sha256, write_json
from stprotein2.neural import CompactResidual
from stprotein2.pipeline import load_bundle, predict_file
from stprotein2.provenance import sources
from stprotein2.smoothing import apply, spatial_residual


LEGACY = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
import joblib, numpy as np, torch
from stprotein2_optimizer_v1.nested import run_nested
from stprotein2.data import Cohort
registry, groups, config, output, cache = map(Path, sys.argv[1:])
torch.set_num_threads(2)
args = SimpleNamespace(registry=str(registry), groups=str(groups), config=str(config),
                       query='query', output=str(output), anchor_cache=str(cache),
                       reuse_anchor_cache=[], device='cpu', cuda_allocator_gib=4.)
run_nested(args)
cohort = Cohort(registry, json.loads(groups.read_text())['groups'])
prep = joblib.load(output/'preprocessing.joblib')
view = cohort.input('query', prep['transform'])
np.savez_compressed(output/'query_view.npz', x=view.x, availability=view.availability,
                    qc=view.qc, coordinates=view.coordinates, barcodes=view.barcodes,
                    genes=np.asarray(view.genes))
assert 'query' not in cohort.protein_reads
'''


NATIVE = r'''
import json, os, sys
from pathlib import Path
research, clean_core, package = (Path(p).resolve() for p in sys.argv[1:4])
class BlockHistoricalPackages:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0].startswith('stprotein2_'):
            raise RuntimeError('Historical package import: ' + fullname)
sys.meta_path.insert(0, BlockHistoricalPackages())
def no_source_checkout(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        if any(p == root or root in p.parents for root in (research, clean_core)):
            raise RuntimeError('Source checkout access: ' + str(p))
sys.addaudithook(no_source_checkout)
import stprotein2
assert package in Path(stprotein2.__file__).resolve().parents
from stprotein2.cli import main
main(sys.argv[4:])
print(json.dumps({'native_package': str(Path(stprotein2.__file__).resolve()),
                  'historical_packages_imported': False, 'checkout_source_access': False}))
'''


def fixture(folder):
    folder.mkdir()
    rng = np.random.default_rng(5207)
    entries, groups = [], {}
    for i, (name, group) in enumerate([('a1', 'a'), ('a2', 'a'), ('b', 'b'), ('c', 'c'),
                                      ('d', 'd'), ('e', 'e'), ('query', 'q'), ('query_mate', 'q')]):
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (11 + i % 3, 10)).astype(np.float32)))
        rna.obs_names = [f'{name}-{j}' for j in range(len(rna))]
        rna.var_names = [f'g{j}' for j in range(10)]
        rna.obsm['spatial'] = np.array([(j // 4, j % 4) for j in range(len(rna))], dtype=float)
        rna.obsm['spatial'][-1] = [100, 100]
        rp, ap = folder / f'{name}_RNA.h5ad', folder / f'{name}_ADT.h5ad'
        rna.write_h5ad(rp)
        if group != 'q':
            protein = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (len(rna), 31)).astype(np.float32)))
            protein.obs_names, protein.var_names = rna.obs_names, PANEL
            protein.write_h5ad(ap)
        entries.append({'name': name, 'rna': str(rp), 'adt': str(ap)})
        groups[name] = group
    registry, group_file = folder / 'registry.json', folder / 'groups.json'
    write_json(registry, entries); write_json(group_file, {'groups': groups})
    return registry, group_file, folder / 'query_RNA.h5ad'


def run(command, cwd, env, log):
    with log.open('w') as stream:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Training compatibility run failed; see ' + str(log))


def compare_block(native, legacy):
    new_protocol = json.loads((native / 'manifest.json').read_text())['protocol']
    old_protocol = json.loads((legacy / 'manifest.json').read_text())['protocol']
    for key in ('reference_identity', 'validation_identity'):
        if new_protocol[key] != old_protocol[key]:
            raise ValueError('Preprocessing, source or target identity differs: ' + key)
    for name, checksum in new_protocol['baseline_sha256'].items():
        if checksum != old_protocol['baseline_signature'][name]['prediction_sha256']:
            raise ValueError('Training baseline arithmetic differs')
    old_saved = torch.load(legacy / 'model.pt', map_location='cpu', weights_only=True)
    expected, _ = CompactResidual.from_research_checkpoint(old_saved)
    actual = torch.load(native / 'residual.pt', map_location='cpu', weights_only=True)['state_dict']
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    for name, field in [('weight', 'anchor.weight'), ('bias', 'anchor.bias')]:
        if digest_array(old_saved['state_dict'][field].numpy()) != new_protocol['anchor'][name]:
            raise ValueError('Independently fitted anchor coefficients differ')
    new_report = json.loads((native / 'training.json').read_text())
    old_report = json.loads((legacy / 'training.json').read_text())
    for key in ('selected_epoch', 'reference_validation_mse'):
        if new_report[key] != old_report[key]:
            raise ValueError('Checkpoint selection differs: ' + key)
    for a, b in zip(new_report['history'], old_report['history'], strict=True):
        for key in ('loss', 'protein', 'last_gradient_norm', 'optimizer_updates', 'reference_validation_mse'):
            if a.get(key) != b.get(key):
                raise ValueError('Training trajectory differs: ' + key)
    return {'selected_epoch': new_report['selected_epoch'], 'retained_weights_max_error': 0.,
            'input_and_baseline_identities_equal': True, 'history_values_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--head', action='append', choices=('own_mlp', 'neighbor_mlp'))
    parser.add_argument('--wheel', type=Path, help='Run native training from an isolated matching wheel')
    args = parser.parse_args()
    research, output = args.research_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh compatibility directory')
    output.mkdir(parents=True)
    registry, groups, rna = fixture(output / 'data')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    env.pop('PYTHONPATH', None)
    torch.set_num_threads(2)
    native_prefix, native_env, native_cwd = [sys.executable, '-m', 'stprotein2'], env, ROOT
    wheel_sha256 = None
    if args.wheel:
        isolated = tempfile.TemporaryDirectory(prefix='stprotein2-wheel-training-')
        atexit.register(isolated.cleanup)
        package = Path(isolated.name) / 'package'; package.mkdir()
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(package)
            identities = {n: sha256(package / n) for n in archive.namelist()
                          if n.startswith('stprotein2/') and n.endswith('.py')}
        if identities != sources():
            raise ValueError('Wheel sources differ from the clean implementation')
        wheel_sha256 = sha256(args.wheel)
        native_prefix = [sys.executable, '-c', NATIVE, str(research), str(ROOT / 'stprotein2'), str(package)]
        native_env, native_cwd = {**env, 'PYTHONPATH': str(package)}, Path(isolated.name)
    heads = args.head or ['own_mlp', 'neighbor_mlp']
    if len(heads) != len(set(heads)):
        raise ValueError('Heads must be unique')
    write_json(output / 'protocol.json', {'scope': 'synthetic software compatibility; not biological model selection',
               'heads': heads, 'sources': 6, 'expected_fits_per_head': 31,
               'epochs': 4, 'warmup_epochs': 1, 'first_checkpoint_epoch': 2, 'seed': 2025,
               'required_tensor_and_training_tolerance': 0., 'query_ADT_absent': True,
               'native_training_from_isolated_wheel': bool(args.wheel), 'wheel_sha256': wheel_sha256,
               'script_sha256': sha256(__file__), 'source_sha256': sources()})
    records = []
    for head in heads:
        model = {'n_genes': 8, 'width': 8, 'hidden': 6, 'members': 2, 'residual_width': 10}
        native_config = output / f'{head}_native.json'
        write_json(native_config, {'model': model, 'train': {'epochs': 4, 'warmup_epochs': 1, 'checkpoint_delay': 1,
                   'validate_every': 1, 'patience': 3, 'seed': 2025}})
        legacy_config = output / f'{head}_legacy.json'
        write_json(legacy_config, {'anchor_cache_policy': 'fit', 'model': {**model, 'residual_head': head,
                   'use_state': False, 'use_niche': False, 'local_operator': 'none'},
                   'train': {'epochs': 4, 'warmup_epochs': 1, 'prototype_ramp_epochs': 1,
                   'validate_every': 1, 'patience': 3, 'seed': 2025, 'lr': .001, 'optimizer_epsilon': 1e-8,
                   'teacher_mode': 'none', 'residual_target': 'in_fit', 'execution_mode': 'full',
                   'threads': 2, 'router_components': 2, 'router_clusters': 2, 'router_per_source': 10}})
        native, legacy = output / f'{head}_native', output / f'{head}_legacy'
        run([*native_prefix, 'train', '--model', head, '--genes', '8', '--config', str(native_config),
             '--registry', str(registry), '--groups', str(groups), '--query', 'query', '--output', str(native)],
            native_cwd, native_env, output / f'{head}_native.log')
        run([sys.executable, '-c', LEGACY, str(registry), str(groups), str(legacy_config), str(legacy),
             str(output / 'legacy_anchor_cache')], research, env, output / f'{head}_legacy.log')
        markers = sorted(native.rglob('fit_complete.json'))
        if len(markers) != 31:
            raise ValueError('All 31 nested fits are required')
        fits = {str(p.parent.relative_to(native)): compare_block(p.parent, legacy / p.parent.relative_to(native))
                for p in markers}
        if json.loads((native / 'epoch_selection.json').read_text()) != json.loads((legacy / 'epoch_selection.json').read_text()):
            raise ValueError('Final epoch aggregation differs')
        new_gates, old_gates = [json.loads((p / 'gate_selection.json').read_text()) for p in (native, legacy)]
        for key in ('gamma_state', 'gamma_local', 'proteins', 'groups', 'thresholds', 'per_protein'):
            if new_gates[key] != old_gates[key]:
                raise ValueError('Reference gate evidence or decision differs: ' + key)
        old_weights = torch.load(legacy / 'model.pt', map_location='cpu', weights_only=True)['state_dict']
        new_weights = torch.load(native / 'residual.pt', map_location='cpu', weights_only=True)['state_dict']
        converted, _ = CompactResidual.from_research_checkpoint(torch.load(legacy / 'model.pt', map_location='cpu', weights_only=True))
        for name, value in converted.state_dict().items():
            torch.testing.assert_close(value, new_weights[name], rtol=0, atol=0)
        anchor = load_bundle(native)
        view = anchor['transform'].transform(ad.read_h5ad(rna), 'query', 'q')
        with np.load(legacy / 'query_view.npz', allow_pickle=False) as old:
            for key in ('x', 'availability', 'qc', 'coordinates', 'barcodes', 'genes'):
                np.testing.assert_array_equal(getattr(view, key), old[key])
        predict_file(native, rna, output / f'{head}_prediction', sample='query', threads=2)
        x = np.where(view.availability > 0, view.x, 0).astype(np.float64)
        raw = (x @ old_weights['anchor.weight'].numpy().astype(float).T + old_weights['anchor.bias'].numpy().astype(float)).astype(np.float32)
        graph = physical_graph(view.coordinates)
        base = apply(raw, spatial_residual(raw, graph), .5)
        with np.load(legacy / 'predictions/query/prediction.npz', allow_pickle=False) as old, \
             np.load(output / f'{head}_prediction/prediction.npz', allow_pickle=False) as new:
            for key in ('state_delta', 'local_delta', 'local_raw', 'barcodes', 'proteins'):
                np.testing.assert_array_equal(new[key], old[key])
            expected = base + old['gamma_state'] * old['state_delta'] + old['gamma_local'] * old['local_delta']
            np.testing.assert_array_equal(new['prediction'], expected)
        if (output / 'data/query_ADT.h5ad').exists() or (output / 'data/query_mate_ADT.h5ad').exists():
            raise ValueError('Compatibility fixture must never have held query proteins')
        records.append({'head': head, 'fits': fits, 'fit_count': len(fits),
                        'gate_decisions_and_evidence_exact': True, 'RNA_only_prediction_max_error': 0.,
                        'native_training_complete_sha256': sha256(native / 'training_complete.json'),
                        'legacy_prediction_complete_sha256': sha256(legacy / 'prediction_complete.json')})
        print(json.dumps({'head': head, 'fits_passed': len(fits), 'maximum_weight_error': 0.}), flush=True)
    write_json(output / 'complete.json', {'status': 'passed', 'observed_unix': time.time(),
               'protocol_sha256': sha256(output / 'protocol.json'), 'heads': records,
               'biological_performance_evidence': False, 'final_architecture_selected': False})


if __name__ == '__main__':
    main()
