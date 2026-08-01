from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse


ALLOWED_DATASETS = {"Mixamo", "Manny", "SMPL", "MetaHuman"}
ALLOWED_METHODS = {"SKAC", "SAN", "R2ET", "UE", "MotionBuilder"}
QUALITY_LAYERS = {"retargeting", "codec_roundtrip", "artist_gold"}


@dataclass(frozen=True)
class Sample:
    raw: dict[str, Any]
    benchmark_root: Path

    @property
    def sample_id(self) -> str:
        return str(self.raw["sample_id"])

    @property
    def split(self) -> str:
        character = "seen_character" if self.raw["character_seen"] else "unseen_character"
        motion = "seen_motion" if self.raw["motion_seen"] else "unseen_motion"
        return f"{character}_{motion}"

    def resolve(self, field: str) -> Path:
        relative = PurePosixPath(self.raw[field])
        return self.benchmark_root / Path(*relative.parts)


def _validate_relative_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"{field} must be a portable relative path without traversal")
    if not path.parts or path.parts[0] != "public_data":
        raise ValueError(f"{field} must be located below public_data")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version", "sample_id", "dataset", "method", "quality_layer",
        "track", "source_character_id", "target_character_id", "motion_id",
        "character_seen", "motion_seen", "joint_count", "root_index",
        "character_height", "prediction_path", "ground_truth_path",
        "skeleton_config_id", "skeleton_config_sha256", "configuration_scope",
        "per_animation_adjustment", "provenance_url", "license_note",
        "geometry_backend",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"missing manifest fields: {', '.join(missing)}")
    if record["schema_version"] != "0.1.0":
        raise ValueError("unsupported schema_version")
    if record["dataset"] not in ALLOWED_DATASETS:
        raise ValueError("dataset is outside the public benchmark allowlist")
    if record["method"] not in ALLOWED_METHODS:
        raise ValueError("method is outside the benchmark allowlist")
    if record["quality_layer"] not in QUALITY_LAYERS:
        raise ValueError("unknown quality_layer")
    if record["track"] not in {"automatic", "artist_gold"}:
        raise ValueError("unknown track")
    if record["track"] == "automatic" and record["per_animation_adjustment"] is not False:
        raise ValueError("automatic track forbids per-animation adjustment")
    if record["track"] == "artist_gold" and record["quality_layer"] != "artist_gold":
        raise ValueError("artist_gold track must use the artist_gold quality layer")
    if record["configuration_scope"] != "skeleton":
        raise ValueError("configuration_scope must be skeleton")
    if not isinstance(record["character_seen"], bool) or not isinstance(record["motion_seen"], bool):
        raise ValueError("seen flags must be booleans")
    if record["dataset"] == "Mixamo" and record["quality_layer"] == "retargeting":
        if record["joint_count"] != 22:
            raise ValueError("primary Mixamo retargeting records require 22 joints")
    if not isinstance(record["joint_count"], int) or record["joint_count"] <= 0:
        raise ValueError("joint_count must be a positive integer")
    if not isinstance(record["root_index"], int) or not 0 <= record["root_index"] < record["joint_count"]:
        raise ValueError("root_index is outside the joint range")
    if not isinstance(record["character_height"], (int, float)) or record["character_height"] <= 0:
        raise ValueError("character_height must be positive")
    _validate_relative_path(record["prediction_path"], "prediction_path")
    _validate_relative_path(record["ground_truth_path"], "ground_truth_path")
    _validate_sha256(record["skeleton_config_sha256"], "skeleton_config_sha256")
    parsed = urlparse(record["provenance_url"])
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("provenance_url must be an HTTPS URL")
    if not isinstance(record["license_note"], str) or not record["license_note"].strip():
        raise ValueError("license_note is required")


def load_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    benchmark_root = path.parent.parent if path.parent.name == "manifests" else path.parent
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            sample_id = str(record["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)
            samples.append(Sample(record, benchmark_root))
    if not samples:
        raise ValueError("manifest has no samples")
    layers = {sample.raw["quality_layer"] for sample in samples}
    tracks = {sample.raw["track"] for sample in samples}
    if len(layers) != 1 or len(tracks) != 1:
        raise ValueError("one evaluation run cannot mix quality layers or tracks")
    return samples


def require_frozen_skeleton_configs(samples: Iterable[Sample]) -> None:
    configs_by_skeleton: dict[tuple[str, str], tuple[str, str]] = {}
    for sample in samples:
        skeleton_key = (
            str(sample.raw["dataset"]),
            str(sample.raw["target_character_id"]),
        )
        config_id = str(sample.raw["skeleton_config_id"])
        config_hash = str(sample.raw["skeleton_config_sha256"])
        config = (config_id, config_hash)
        previous = configs_by_skeleton.setdefault(skeleton_key, config)
        if previous != config:
            raise ValueError(
                "target skeleton configuration changed within run: "
                f"{sample.raw['target_character_id']}"
            )
