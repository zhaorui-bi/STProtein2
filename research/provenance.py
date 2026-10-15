"""Freeze repository experiments without coupling core artifacts to research code."""
from pathlib import Path

from stprotein2.io import freeze_run as freeze_core_run, sha256
from stprotein2.provenance import sources as core_sources

ROOT = Path(__file__).resolve().parents[1]


def sources():
    files = list((ROOT / 'research').rglob('*.py'))
    files += list((ROOT / 'evaluation').rglob('*.py'))
    files += list((ROOT / 'docs/history/protocols').glob('*.md'))
    return {**core_sources(), **{
        str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)
    }}


def freeze_run(output, payload):
    return freeze_core_run(output, {**payload, 'research_source_sha256': sources()})


def snapshot_sources(output):
    destination = Path(output) / 'source_snapshot'
    for relative, digest in sources().items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise RuntimeError('Frozen research source snapshot differs')
        else:
            target.write_bytes((ROOT / relative).read_bytes())
