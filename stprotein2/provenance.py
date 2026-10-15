"""One source-identity and snapshot implementation for the unified package."""
from pathlib import Path

from .io import sha256
from .paths import PACKAGE_ROOT, SOURCE_ROOT


def sources():
    files = list(PACKAGE_ROOT.rglob('*.py'))
    files += list((PACKAGE_ROOT / 'resources').rglob('*.md'))
    return {str(p.relative_to(SOURCE_ROOT)): sha256(p) for p in sorted(files)}


def snapshot_sources(output):
    destination = Path(output) / 'source_snapshot'
    for relative, digest in sources().items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise RuntimeError('Frozen STProtein2 source snapshot differs')
        else:
            target.write_bytes((SOURCE_ROOT / relative).read_bytes())
