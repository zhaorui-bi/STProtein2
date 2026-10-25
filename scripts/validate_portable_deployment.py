"""Replay two real verified checkpoints from a wheel outside both checkouts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stprotein2.io import sha256, write_json
from stprotein2.provenance import sources


ISOLATED = r'''
import json, os, sys
from pathlib import Path
research, wheel, bundle, rna, output = (Path(p).resolve() for p in sys.argv[1:])
class BlockHistoricalPackages:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0].startswith('stprotein2_'):
            raise RuntimeError('Historical package import: ' + fullname)
sys.meta_path.insert(0, BlockHistoricalPackages())
def no_research_files(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        if p == research or research in p.parents:
            raise RuntimeError('Historical file access: ' + str(p))
sys.addaudithook(no_research_files)
import anndata as ad
reads = []
real_read = ad.read_h5ad
def rna_only(path, *args, **kwargs):
    if Path(path).resolve() != rna.resolve():
        raise RuntimeError('Expression access outside the query RNA')
    reads.append(str(Path(path).resolve()))
    return real_read(path, *args, **kwargs)
ad.read_h5ad = rna_only
from stprotein2.pipeline import predict_file
import stprotein2
assert wheel in Path(stprotein2.__file__).resolve().parents
predict_file(bundle, rna, output, sample='Breast', threads=2)
assert reads == [str(rna.resolve())]
assert not any(k.startswith('stprotein2_') for k in sys.modules)
print(json.dumps({'package': str(Path(stprotein2.__file__).resolve()),
                  'RNA_files_read': len(reads), 'research_file_access': False,
                  'historical_packages_imported': False}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    research, output = args.research_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh validation directory')
    entries = json.loads((research / 'data/extended_development_20261013/pairs_raw.json').read_text())
    rna = Path(next(a for a in entries if a['name'] == 'Breast')['rna'])
    campaign = research / 'runs/icml_optimizer_training_20261021_v1'
    cases = [('own_mlp', 0), ('neighbor_mlp', 3)]
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', {'scope': 'development deployment replay; no new fit or selection',
               'script_sha256': sha256(__file__), 'source_sha256': sources(),
               'query': 'Breast', 'query_RNA_sha256': sha256(rna), 'seeds': [2025],
               'atol': 1e-5, 'rtol': 2e-5, 'measured_protein_read_or_hashed': False})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    env.pop('PYTHONPATH', None)
    for head, index in cases:
        run = campaign / f'stable/{head}_eps1e-08/seed_2025/Breast'
        with (output / f'{head}_export.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'stprotein2', 'export', '--run', str(run),
                            '--verification', str(campaign / f'independent_verification/{index}'),
                            '--output', str(output / f'{head}_bundle')], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / 'wheel_build.log').open('w') as log:
        subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(output / 'dist')], cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    wheel, = (output / 'dist').glob('*.whl')
    records = []
    with tempfile.TemporaryDirectory(prefix='stprotein2-wheel-replay-') as temporary:
        isolated = Path(temporary)
        package = isolated / 'package'; package.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(package)
            wheel_sources = {n: sha256(package / n) for n in archive.namelist()
                             if n.startswith('stprotein2/') and n.endswith('.py')}
        if wheel_sources != sources():
            raise ValueError('Wheel source identities differ from the exported implementation')
        copied_rna = isolated / 'query_RNA.h5ad'; shutil.copyfile(rna, copied_rna)
        for head, index in cases:
            bundle = isolated / f'{head}_bundle'
            shutil.copytree(output / f'{head}_bundle', bundle)
            prediction = isolated / f'{head}_prediction'
            completed = subprocess.run([sys.executable, '-c', ISOLATED, str(research), str(package),
                                       str(bundle), str(copied_rna), str(prediction)], cwd=isolated,
                                      env={**env, 'PYTHONPATH': str(package)}, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (output / f'{head}_isolated.log').write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError('Isolated replay failed; see ' + str(output / f'{head}_isolated.log'))
            runtime = json.loads(completed.stdout.strip().splitlines()[-1])
            source_prediction = campaign / f'stable/{head}_eps1e-08/seed_2025/Breast/predictions/Breast/prediction.npz'
            errors = {}
            with np.load(prediction / 'prediction.npz', allow_pickle=False) as new, np.load(source_prediction, allow_pickle=False) as old:
                for key in ('barcodes', 'proteins', 'gamma_state', 'gamma_local'):
                    np.testing.assert_array_equal(new[key], old[key])
                for key in ('prediction', 'base', 'rna_base', 'state_delta', 'local_delta', 'local_raw'):
                    np.testing.assert_allclose(new[key], old[key], atol=1e-5, rtol=2e-5)
                    errors[key] = float(np.max(np.abs(new[key].astype(float) - old[key])))
                shape = list(new['prediction'].shape)
                active = 'state_delta' if head == 'own_mlp' else 'local_delta'
                if np.max(np.abs(new[active])) == 0 or not np.any(new['gamma_state'] + new['gamma_local']):
                    raise ValueError('Real replay must exercise an active nonzero correction')
            shutil.move(str(prediction), output / f'{head}_prediction')
            records.append({'head': head, 'shape': shape, 'maximum_absolute_errors': errors,
                            'source_prediction_sha256': sha256(source_prediction),
                            'bundle_complete_sha256': sha256(bundle / 'bundle_complete.json'),
                            'verification_report_sha256': sha256(campaign / f'independent_verification/{index}/report.json'),
                            'isolated_runtime': runtime})
    write_json(output / 'complete.json', {'status': 'passed', 'observed_unix': time.time(),
               'protocol_sha256': sha256(output / 'protocol.json'), 'wheel_sha256': sha256(wheel),
               'wheel_source_sha256': wheel_sources, 'cases': records,
               'measured_protein_read_or_hashed': False, 'neural_training_API_validated': False,
               'final_architecture_selected': False})
    print(json.dumps({'status': 'passed', 'cases': len(records), 'output': str(output)}))


if __name__ == '__main__':
    main()
