"""Core/repository boundaries, public commands and source-integrity contracts."""
import ast
import importlib
from pathlib import Path

import pytest

from stprotein2 import cli
from stprotein2.io import source_manifest, sha256
from stprotein2.paths import PACKAGE_ROOT, SOURCE_ROOT


def test_supported_modules_resolve_without_repository_tool_imports():
    for name in ('models.spatial', 'training.nested', 'training.trainer',
                 'inference.bundle',
                 'execution.memory', 'execution.planner'):
        module = importlib.import_module('stprotein2.' + name)
        assert Path(module.__file__).resolve().is_relative_to(PACKAGE_ROOT)
    for path in PACKAGE_ROOT.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                assert not name.startswith('stprotein2_'), (path, name)
                assert name.split('.')[0] not in {'research', 'evaluation'}, (path, name)
    for name in ('research', 'evaluation', 'resources'):
        assert not (PACKAGE_ROOT / name).exists()


def test_public_commands_forward_to_current_stages(monkeypatch):
    from stprotein2.training import __main__ as training
    from stprotein2.inference import __main__ as inference
    calls = []
    monkeypatch.setattr(training, 'main', lambda args: calls.append(('training', args)))
    monkeypatch.setattr(inference, 'main', lambda args: calls.append(('inference', args)))
    cli.main(['train', '--config', 'current.json'])
    cli.main(['prepare', '--config', 'current.json'])
    cli.main(['export', '--run', 'finished'])
    cli.main(['predict', '--rna', 'query.h5ad'])
    assert calls == [
        ('training', ['run', '--config', 'current.json']),
        ('training', ['prepare', '--config', 'current.json']),
        ('inference', ['export', '--run', 'finished']),
        ('inference', ['predict', '--rna', 'query.h5ad']),
    ]


def test_measured_adt_scoring_has_its_own_repository_entry_point(monkeypatch):
    from evaluation import score
    calls = []
    monkeypatch.setattr(score, 'score', lambda registry, output, **kwargs: calls.append((registry, output, kwargs)))
    score.main(['--registry', 'registry.json', '--output', 'finished'])
    assert calls == [('registry.json', 'finished', {'extended': False, 'cluster_counts': (4, 6, 8), 'report_dir': None})]
    with pytest.raises(SystemExit) as error:
        cli.main(['score'])
    assert error.value.code == 2


def test_source_manifest_covers_relocated_model_and_training_code():
    manifest = source_manifest()
    required = ('models/residual.py', 'models/spatial.py', 'models/sheaf.py',
                'models/prototypes.py', 'training/trainer.py', 'training/nested.py',
                'inference/bundle.py')
    for relative in required:
        name = 'stprotein2/' + relative
        assert manifest[name] == sha256(SOURCE_ROOT / name)
    assert all(name.startswith('stprotein2/') for name in manifest)
    assert all(name.endswith('.py') for name in manifest)


def test_research_identity_is_recorded_separately_from_core(tmp_path, monkeypatch):
    from research import provenance
    from stprotein2.io import freeze_run
    core = source_manifest()
    experiment = provenance.sources()
    assert all(experiment[name] == digest for name, digest in core.items())
    assert any(name.startswith('research/') for name in experiment)
    assert any(name.startswith('evaluation/') for name in experiment)
    assert any(name.startswith('docs/history/protocols/') for name in experiment)
    frozen = provenance.freeze_run(tmp_path / 'research', {'stage': 'test'})
    assert frozen['source_sha256'] == core
    assert frozen['protocol']['research_source_sha256'] == experiment
    freeze_run(tmp_path / 'core', {'stage': 'test'})
    monkeypatch.setattr(provenance, 'sources', lambda: {**experiment, 'research/changed.py': 'changed'})
    with pytest.raises(RuntimeError, match='changed'):
        provenance.freeze_run(tmp_path / 'research', {'stage': 'test'})
    assert freeze_run(tmp_path / 'core', {'stage': 'test'})['source_sha256'] == core
