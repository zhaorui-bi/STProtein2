"""Run installed-package checks and a complete demo outside the source checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    if work.is_relative_to(root):
        parser.error('--work-dir must be outside the source checkout')
    work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)

    def run(*arguments):
        subprocess.run([sys.executable, *map(str, arguments)], cwd=work, env=env, check=True)

    run('-c', '''
import importlib.abc
import json
from pathlib import Path
import sys

class RejectRepositoryTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'evaluation', 'research'}:
            raise AssertionError('Core imported repository tool: ' + fullname)

sys.meta_path.insert(0, RejectRepositoryTools())
from stprotein2 import STProtein2, ModelConfig, export_run, load_bundle, predict_file
from stprotein2.paths import PACKAGE_ROOT
from stprotein2.provenance import sources
from stprotein2.io import sha256
from stprotein2 import cli
root = Path(sys.argv[1]).resolve()
assert not PACKAGE_ROOT.is_relative_to(root / 'stprotein2'), PACKAGE_ROOT
for name in ('evaluation', 'research', 'resources'):
    assert not (PACKAGE_ROOT / name).exists(), name
expected = {str(p.relative_to(root)): sha256(p) for p in sorted((root/'stprotein2').rglob('*.py'))}
assert sources() == expected
for command in ([], ['train', '--help'], ['prepare', '--help'], ['export', '--help'], ['predict', '--help']):
    try:
        cli.main(command)
    except SystemExit as error:
        assert error.code == 0
Path('distribution_validation.json').write_text(json.dumps({
    'outside_checkout': True, 'repository_tool_imports_blocked': True,
    'core_only_wheel': True, 'repository_and_wheel_source_manifests_equal': True,
    'core_source_files': len(expected),
}, indent=2) + '\\n')
''', root)
    run(root / 'examples/make_demo.py', '--output', 'data')
    run('-m', 'stprotein2', 'train', '--registry', 'data/registry.json', '--groups', 'data/groups.json',
        '--config', root / 'configs/demo.json', '--query', 'query', '--output', 'run',
        '--anchor-cache', 'anchors', '--device', 'cpu')
    run('-m', 'stprotein2', 'export', '--run', 'run', '--output', 'bundle')
    run('-m', 'stprotein2', 'predict', '--bundle', 'bundle', '--rna', 'data/query_rna.h5ad',
        '--sample', 'query', '--output', 'prediction', '--device', 'cpu')
    run('-c', '''
import json
from pathlib import Path
import numpy as np
from stprotein2.inference import load_bundle
from stprotein2.training.config import BalancedTrainConfig

assert not Path('data/query_adt.h5ad').exists()
with np.load('run/predictions/query/prediction.npz') as source, np.load('prediction/prediction.npz') as deployed:
    for key in ('prediction', 'base', 'rna_base', 'state_delta', 'local_delta', 'spatial_basis_delta'):
        np.testing.assert_array_equal(source[key], deployed[key])
    assert deployed['prediction'].shape == (16, 31)
    assert np.isfinite(deployed['prediction']).all()
metadata, prep, model, cfg = load_bundle('bundle')
assert isinstance(cfg, BalancedTrainConfig)
assert cfg.teacher_mode == 'none' and cfg.residual_target == 'in_fit'
result = json.loads(Path('distribution_validation.json').read_text())
result.update(actual_nested_training=True, query_adt_absent=True,
              export_prediction_component_exact_parity=True, shape=[16, 31], runtime=metadata['runtime'])
Path('distribution_validation.json').write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
''')


if __name__ == '__main__':
    main()
