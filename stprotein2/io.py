"""Atomic artifacts and checksums for the minimal reference baseline."""
import hashlib
import json
import os
import platform
from pathlib import Path

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


def freeze_run(output,payload):
    from . import __version__
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(json.dumps({'protocol':payload,'source_sha256':source_manifest(),
        'python':platform.python_version(),'version':__version__},allow_nan=False))
    path=output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('Run code/configuration/inputs changed; choose a fresh directory')
    if not path.exists():write_json(path,manifest)
    return manifest
