"""Public commands and source-integrity contracts of the unified package."""
import ast
import importlib
from pathlib import Path

from stprotein2 import cli
from stprotein2.io import source_manifest, sha256
from stprotein2.paths import PACKAGE_ROOT, SOURCE_ROOT


def test_supported_modules_resolve_inside_one_package_without_research_imports():
    for name in ('models.spatial', 'training.nested', 'training.trainer',
                 'inference.bundle', 'evaluation.score', 'evaluation.spatial',
                 'execution.memory', 'execution.planner'):
        module = importlib.import_module('stprotein2.' + name)
        assert Path(module.__file__).resolve().is_relative_to(PACKAGE_ROOT)
    for path in PACKAGE_ROOT.rglob('*.py'):
        if 'research' in path.relative_to(PACKAGE_ROOT).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                assert not name.startswith('stprotein2_'), (path, name)
                assert not name.startswith('stprotein2.research'), (path, name)


def test_public_commands_forward_to_current_stages(monkeypatch):
    from stprotein2.training import __main__ as training
    from stprotein2.inference import __main__ as inference
    from stprotein2.evaluation import score
    calls = []
    monkeypatch.setattr(training, 'main', lambda args: calls.append(('training', args)))
    monkeypatch.setattr(inference, 'main', lambda args: calls.append(('inference', args)))
    monkeypatch.setattr(score, 'main', lambda args: calls.append(('score', args)))
    cli.main(['train', '--config', 'current.json'])
    cli.main(['prepare', '--config', 'current.json'])
    cli.main(['export', '--run', 'finished'])
    cli.main(['predict', '--rna', 'query.h5ad'])
    cli.main(['score', '--output', 'finished'])
    assert calls == [
        ('training', ['run', '--config', 'current.json']),
        ('training', ['prepare', '--config', 'current.json']),
        ('inference', ['export', '--run', 'finished']),
        ('inference', ['predict', '--rna', 'query.h5ad']),
        ('score', ['--output', 'finished']),
    ]


def test_source_manifest_covers_relocated_model_and_training_code():
    manifest = source_manifest()
    required = ('models/residual.py', 'models/spatial.py', 'models/sheaf.py',
                'models/prototypes.py', 'training/trainer.py', 'training/nested.py',
                'inference/bundle.py', 'resources/docs/LATEST_BENCHMARK_PROTOCOL_20261013.md')
    for relative in required:
        name = 'stprotein2/' + relative
        assert manifest[name] == sha256(SOURCE_ROOT / name)
    assert all(name.startswith('stprotein2/') for name in manifest)
