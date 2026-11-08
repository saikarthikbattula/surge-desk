"""Completion manifests: data are published first and verified before reuse."""
from pathlib import Path

from . import config, fileio
from .provenance import digest, file_hash


def complete(path, identity, outputs):
    fileio.write_json(path, {"schema": config.RESEARCH_SCHEMA, "identity": digest(identity), "inputs": identity,
                             "outputs": {Path(p).name: file_hash(p) for p in outputs}})


def valid(path, identity, outputs):
    try:
        manifest = fileio.read_json(path)
        return (manifest["schema"] == config.RESEARCH_SCHEMA
                and manifest["identity"] == digest(identity)
                and manifest["outputs"] == {Path(p).name: file_hash(p) for p in outputs})
    except (OSError, ValueError, KeyError, TypeError):
        return False


def verify(path):
    """Validate a managed directory when opening it without producer inputs."""
    path = Path(path)
    manifest = fileio.read_json(path)
    if manifest.get("schema") != config.RESEARCH_SCHEMA:
        raise ValueError("Unsupported artifact manifest; rebuild stage")
    for name, expected in manifest["outputs"].items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent.resolve()) or file_hash(target) != expected:
            raise ValueError(f"Artifact {name} changed; rebuild stage")
    return manifest
