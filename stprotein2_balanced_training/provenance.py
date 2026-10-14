"""Freeze the new candidate and its unchanged upstream dependencies."""
from pathlib import Path
from stprotein2.io import sha256
from stprotein2_teacher_ablation_v2.provenance import sources as upstream_sources


def sources():
    root = Path(__file__).resolve().parents[1]
    result = upstream_sources()
    for package in ("stprotein2_balanced_initialization", "stprotein2_execution", "stprotein2_balanced_training"):
        for path in sorted((root/package).glob("*.py")):
            result[str(path.relative_to(root))] = sha256(path)
    for relative in ("stprotein2_resources/docs/BALANCED_INITIALIZATION_PROTOCOL_20261013.md", "stprotein2_resources/docs/BALANCED_TRAINING_PROTOCOL_20261013.md"):
        result[relative] = sha256(root/relative)
    return result


def snapshot_sources(output):
    root = Path(__file__).resolve().parents[1]
    for relative in sources():
        target = Path(output)/"source_snapshot"/relative
        source = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError("Frozen balanced-training source snapshot differs")
        if not target.exists():
            target.write_bytes(source.read_bytes())
