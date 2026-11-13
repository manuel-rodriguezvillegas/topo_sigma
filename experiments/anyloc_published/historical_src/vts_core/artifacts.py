"""Content identities and atomic publication of experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def file_digest(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


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


def atomic_json(path: str | Path, value: object) -> None:
    atomic_bytes(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )


def completed_run(directory: str | Path, fingerprint: str | None = None) -> dict | None:
    """Verify the entire completed mapping artifact set, never file mtimes."""
    directory = Path(directory)
    try:
        marker = json.loads((directory / "completion.json").read_text())
        if fingerprint is not None and marker["fingerprint"] != fingerprint:
            return None
        if not marker["artifacts"] or marker["frames"] <= 0:
            return None
        for name, digest in marker["artifacts"].items():
            if Path(name).name != name or file_digest(directory / name) != digest:
                return None
        return marker
    except (OSError, ValueError, KeyError, TypeError):
        return None


def json_safe(value: object) -> object:
    """Represent undefined diagnostic numbers as JSON null, never NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
