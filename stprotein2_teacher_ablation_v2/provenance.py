"""Freeze every dependency of the new protocol without mutating old jobs."""
from pathlib import Path

from stprotein2.io import sha256


PACKAGES = ("stprotein2", "stprotein2_anchor", "stprotein2_crossfit", "stprotein2_smoothing", "stprotein2_partition", "stprotein2_partition_cost", "stprotein2_spatial_anchor", "stprotein2_teacher_ablation", "stprotein2_teacher_ablation_v2", "stprotein2_extended_baselines")


def sources():
    root = Path(__file__).resolve().parents[1]
    result = {f"{package}/{p.name}": sha256(p) for package in PACKAGES
              for p in sorted((root/package).glob("*.py"))}
    for path in ("stprotein2_resources/docs/TEACHER_ABLATION_PROTOCOL_20261013.md", "stprotein2_resources/scripts/score_extended_baseline.py", "stprotein2_resources/docs/TEACHER_CACHE_IDENTITY_PROTOCOL_20261013.md"):
        result[path] = sha256(root / path)
    return result


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
