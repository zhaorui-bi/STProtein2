"""Record cohort-feasibility adapter plus unchanged v2.1 model dependencies."""
from pathlib import Path
from stprotein2.io import sha256
from stprotein2_balanced_training.provenance import sources as upstream_sources


def sources():
    root=Path(__file__).resolve().parents[1]
    result=upstream_sources()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        result[str(p.relative_to(root))]=sha256(p)
    p=root/'stprotein2_resources/docs/LATEST_BENCHMARK_PROTOCOL_20261013.md'
    result[str(p.relative_to(root))]=sha256(p)
    return result


def snapshot_sources(output):
    root=Path(__file__).resolve().parents[1]
    for relative,checksum in sources().items():
        source=root/relative;target=Path(output)/'source_snapshot'/relative
        if target.exists() and sha256(target)!=checksum:
            raise RuntimeError('Frozen latest-benchmark source changed')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
