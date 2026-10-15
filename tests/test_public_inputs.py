"""Invalid public settings fail before data access; registries remain portable."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stprotein2 import __version__
from stprotein2.config import ModelConfig, TrainConfig, from_dict
from stprotein2.data import Cohort
from stprotein2.training.config import BalancedTrainConfig


@pytest.mark.parametrize('cls, settings', [
    (ModelConfig, {'width': 4.5}),
    (ModelConfig, {'members': True}),
    (ModelConfig, {'prototype_temperature': float('nan')}),
    (ModelConfig, {'use_state': 'false'}),
    (TrainConfig, {'epochs': 30.5}),
    (TrainConfig, {'lr': float('nan')}),
    (TrainConfig, {'weight_decay': float('inf')}),
    (TrainConfig, {'clip_grad': 0}),
    (TrainConfig, {'base_weight': -1}),
    (TrainConfig, {'teacher_temp_end': 0}),
    (TrainConfig, {'center_momentum': 1}),
    (TrainConfig, {'seed': -1}),
    (TrainConfig, {'prototype_ramp_epochs': -1}),
    (TrainConfig, {'graph_k': True}),
])
def test_invalid_numeric_settings_are_rejected_at_construction(cls, settings):
    with pytest.raises(ValueError):
        cls(**settings)


def test_shipped_configs_are_valid_and_round_trip_without_numerical_changes():
    for path in (Path(__file__).resolve().parents[1] / 'configs').glob('*.json'):
        values = json.loads(path.read_text())
        for cls, key in ((ModelConfig, 'model'), (BalancedTrainConfig, 'train')):
            config = from_dict(cls, values[key])
            assert from_dict(cls, config.to_dict()) == config


def test_registry_relative_paths_do_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    registry = tmp_path / 'cohort/registry.json'
    registry.parent.mkdir()
    registry.write_text(json.dumps([{'name': 'query', 'rna': 'input/query.h5ad', 'adt': 'missing_query_adt.h5ad'}]))
    monkeypatch.chdir(tmp_path)
    cohort = Cohort(registry)
    assert cohort.specs['query']['rna'] == str(registry.parent / 'input/query.h5ad')
    assert cohort.specs['query']['adt'] == str(registry.parent / 'missing_query_adt.h5ad')
    assert not cohort.protein_reads


@pytest.mark.parametrize('entries, groups', [
    ([], None),
    ([{'name': 'query'}], None),
    ([{'name': '../query', 'rna': 'a', 'adt': 'b'}], None),
    ([{'name': 'query', 'rna': '', 'adt': 'b'}], None),
    ([{'name': 'query', 'rna': 'a', 'adt': 'b'}], {}),
    ([{'name': 'query', 'rna': 'a', 'adt': 'b'}], {'query': 1}),
])
def test_invalid_registry_contracts_fail_before_opening_data(tmp_path, entries, groups):
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps(entries))
    with pytest.raises(ValueError):
        Cohort(registry, groups)


def test_version_command_needs_no_scientific_imports():
    code = '''
import importlib.abc
import sys
class RejectScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'anndata'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectScientificImports())
from stprotein2.cli import main
main(['--version'])
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == f'stprotein2 {__version__}'
