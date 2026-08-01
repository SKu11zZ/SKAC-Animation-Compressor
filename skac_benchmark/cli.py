from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import load_manifest, require_frozen_skeleton_configs
from .metrics import evaluate_motion


SPLITS = (
    "seen_character_seen_motion",
    "seen_character_unseen_motion",
    "unseen_character_seen_motion",
    "unseen_character_unseen_motion",
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _mean_present(records: list[dict[str, Any]], field: str) -> float | None:
    values = [record[field] for record in records if record[field] is not None]
    return float(np.mean(values)) if values else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_fields = [
        "global_mse_raw", "global_mse_x1e3", "local_mse_raw",
        "local_mse_x1e3", "san_paper_position_error_x1e3",
        "san_official_code_mse_raw", "san_official_code_mse_x1e3",
        "penetration_percent", "contact_distance_cm",
    ]
    return {
        "sample_count": len(records),
        **{field: _mean_present(records, field) for field in metric_fields},
    }


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    samples = load_manifest(manifest_path)
    require_frozen_skeleton_configs(samples)
    records: list[dict[str, Any]] = []
    for sample in samples:
        prediction = _load_npz(sample.resolve("prediction_path"))
        truth = _load_npz(sample.resolve("ground_truth_path"))
        if "positions" not in prediction or "positions" not in truth:
            raise ValueError(f"{sample.sample_id}: both NPZ files require positions")
        if prediction["positions"].shape[1] != sample.raw["joint_count"]:
            raise ValueError(f"{sample.sample_id}: joint_count does not match prediction")
        metric = evaluate_motion(
            prediction["positions"],
            truth["positions"],
            float(sample.raw["character_height"]),
            int(sample.raw["root_index"]),
            prediction.get("penetrated_limb_vertices"),
            prediction.get("total_limb_vertices"),
            prediction.get("hand_body_distances_cm"),
        )
        records.append(
            {
                "sample_id": sample.sample_id,
                "method": sample.raw["method"],
                "split": sample.split,
                **metric.to_dict(),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return {
        "schema_version": "0.1.0",
        "quality_layer": samples[0].raw["quality_layer"],
        "track": samples[0].raw["track"],
        "groups": {name: _aggregate(grouped.get(name, [])) for name in SPLITS},
        "overall": _aggregate(records),
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="skac-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("manifest", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_manifest(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
