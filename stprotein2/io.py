"""Atomic local artifacts and explicit code/input provenance."""
import hashlib
import json
from pathlib import Path
import platform
import os

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temp); temp.replace(path)


def source_manifest():
    from .provenance import sources
    return sources()


def verify_marker(output, filename):
    """Verify a completed run and every artifact bound by its marker."""
    output = Path(output)
    marker = json.loads((output / filename).read_text())
    if sha256(output / 'manifest.json') != marker['manifest_sha256']:
        raise ValueError('Completed manifest identity changed')
    for relative, checksum in marker['artifacts'].items():
        if sha256(output / relative) != checksum:
            raise ValueError('Completed artifact identity changed: ' + relative)
    return marker


def freeze_run(output, payload):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": payload, "source_sha256": source_manifest(),
                "python": platform.python_version(), "torch": torch.__version__}
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Run code/configuration/inputs changed; choose a fresh run directory")
    if not path.exists():
        write_json(path, manifest)
    return manifest

