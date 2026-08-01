from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Skeleton


RUNTIME_SKELETON_SCHEMA = "skac.runtime_skeleton"
RUNTIME_SKELETON_VERSION = "1.0.0"


def runtime_skeleton_dict(skeleton: Skeleton) -> dict[str, Any]:
    """Return the small, hash-pinned target description consumed by the native runtime."""
    return {
        "schema": RUNTIME_SKELETON_SCHEMA,
        "schema_version": RUNTIME_SKELETON_VERSION,
        "skeleton_sha256": skeleton.signature(),
        "skeleton": skeleton.to_dict(),
    }


def runtime_skeleton_bytes(skeleton: Skeleton) -> bytes:
    return json.dumps(
        runtime_skeleton_dict(skeleton),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def save_runtime_skeleton(path: Path, skeleton: Skeleton) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runtime_skeleton_dict(skeleton), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
