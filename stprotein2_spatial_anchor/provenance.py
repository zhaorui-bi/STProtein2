"""Freeze every dependency of the new protocol without mutating old jobs."""
from pathlib import Path

from stprotein2.io import sha256


PACKAGES = ("stprotein2", "stprotein2_anchor", "stprotein2_crossfit", "stprotein2_smoothing", "stprotein2_partition", "stprotein2_partition_cost", "stprotein2_spatial_anchor")


def sources():
    root = Path(__file__).resolve().parents[1]
    return {f"{package}/{p.name}": sha256(p) for package in PACKAGES
            for p in sorted((root/package).glob("*.py"))}


def snapshot_sources(output):
    root = Path(__file__).resolve().parents[1]
    for relative in sources():
        target = Path(output)/"source_snapshot"/relative
        source = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError("Frozen residual-protocol source snapshot differs")
        if not target.exists():
            target.write_bytes(source.read_bytes())
