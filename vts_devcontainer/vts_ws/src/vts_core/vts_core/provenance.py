"""Configuration loading, content hashes and atomic artifact publication.

Every run records one fingerprint: the hash of the package source, its
parameters and the manifests of its input sessions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "toposigma.yaml"
WORKSPACE = Path(os.environ.get("TOPOSIGMA_WORKSPACE", PACKAGE_DIR.parents[2]))


def file_digest(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=str).encode()).hexdigest()


def code_identity() -> str:
    """Hash of every source file of the package (not of installed copies' paths)."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def packages() -> dict[str, str]:
    out = {}
    for name in ("numpy", "scipy", "torch", "transformers"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


def atomic_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_safe(value: object) -> object:
    """Undefined numbers become null; numpy scalars and arrays become plain JSON."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path: str | Path, value: object) -> None:
    atomic_bytes(path, json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(*overrides: str | Path | dict | None) -> dict:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for override in overrides:
        if override is None:
            continue
        config = _merge(config, override if isinstance(override, dict) else yaml.safe_load(Path(override).read_text()))
    return config


def resolve(config: dict, key: str) -> Path:
    """``store_root`` / ``output_root``, honouring TOPOSIGMA_* environment overrides."""
    path = Path(os.environ.get(f"TOPOSIGMA_{key.upper()}", config[key]))
    return path if path.is_absolute() else WORKSPACE / path


def run_record(command: str, parameters: dict, inputs: dict[str, str]) -> dict:
    record = {"command": command, "code_sha256": code_identity(), "parameters": parameters,
              "inputs": inputs, "packages": packages()}
    record["fingerprint"] = identity(record)
    return record
