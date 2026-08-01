from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score saved SAN BVH outputs with the paper's position-error formula."
    )
    parser.add_argument("--retargeting-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def score_directory(
    bvh_module: Any,
    animation_module: Any,
    bvh_file_type: Any,
    standard_root: Path,
    target_dir: Path,
) -> dict[str, float | int]:
    target = target_dir.name
    height = float(bvh_file_type(str(standard_root / f"{target}.bvh")).get_height())
    predictions = sorted(
        path for path in target_dir.glob("*.bvh") if not path.name.endswith("_gt.bvh")
    )
    errors: list[float] = []
    for prediction in predictions:
        truth = prediction.with_name(f"{prediction.stem}_gt.bvh")
        if not truth.is_file():
            raise FileNotFoundError(truth)
        animation, names, _ = bvh_module.load(str(prediction))
        reference, _, _ = bvh_module.load(str(truth))
        indices = [index for index, name in enumerate(names) if "virtual" not in name]
        positions = animation_module.positions_global(animation)[:, indices, :]
        reference_positions = animation_module.positions_global(reference)[:, indices, :]
        if positions.shape != reference_positions.shape:
            raise ValueError(f"shape mismatch for {prediction}")
        distances = np.linalg.norm(positions - reference_positions, axis=-1)
        errors.append(float(np.mean(distances) / height))
    if not errors:
        raise ValueError(f"no predictions in {target_dir}")
    raw = float(np.mean(errors))
    return {
        "sample_count": len(errors),
        "position_error_raw": raw,
        "position_error_x1e3": raw * 1_000.0,
    }


def mean_detail(
    details: dict[str, dict[str, float | int]], field: str = "position_error_raw"
) -> float:
    return float(np.mean([float(value[field]) for value in details.values()]))


def main() -> None:
    args = parse_args()
    retargeting_root = args.retargeting_root.resolve()
    sys.path.insert(0, str(retargeting_root))
    sys.path.insert(0, str(retargeting_root.parent / "utils"))
    import Animation  # type: ignore[import-not-found]
    import BVH  # type: ignore[import-not-found]
    from datasets.bvh_parser import BVH_file  # type: ignore[import-not-found]

    results_root = args.results_root.resolve()
    standard_root = args.reference_root.resolve()

    cross_details: dict[str, dict[str, float | int]] = {}
    for target_dir in sorted(
        path for path in (results_root / "cross_structure").iterdir() if path.is_dir()
    ):
        cross_details[target_dir.name] = score_directory(
            BVH, Animation, BVH_file, standard_root, target_dir
        )

    intra_details: dict[str, dict[str, dict[str, float | int]]] = {}
    non_identity_values: list[float] = []
    all_values: list[float] = []
    for source_dir in sorted(
        path for path in (results_root / "intra_structure").iterdir() if path.is_dir()
    ):
        source_name = source_dir.name.removeprefix("from_")
        source_scores: dict[str, dict[str, float | int]] = {}
        for target_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            score = score_directory(BVH, Animation, BVH_file, standard_root, target_dir)
            source_scores[target_dir.name] = score
            value = float(score["position_error_raw"])
            all_values.append(value)
            if source_name != target_dir.name:
                non_identity_values.append(value)
        intra_details[source_dir.name] = source_scores

    if not non_identity_values:
        raise ValueError("no non-identity intra-structure source-target pairs found")
    cross_raw = mean_detail(cross_details)
    intra_raw = float(np.mean(non_identity_values))
    intra_all_raw = float(np.mean(all_values))
    report = {
        "report_type": "SAN_paper_formula_on_official_outputs",
        "method": "SAN",
        "cross_structure": {
            "position_error_raw": cross_raw,
            "position_error_x1e3": cross_raw * 1_000.0,
            "target_details": cross_details,
        },
        "intra_structure": {
            "aggregation": "non-identity source-target pairs only",
            "source_target_pair_count": len(non_identity_values),
            "position_error_raw": intra_raw,
            "position_error_x1e3": intra_raw * 1_000.0,
            "diagnostic_including_identity_pairs": {
                "source_target_pair_count": len(all_values),
                "position_error_raw": intra_all_raw,
                "position_error_x1e3": intra_all_raw * 1_000.0,
            },
            "source_target_details": intra_details,
        },
        "metric_definition": (
            "Mean Euclidean global-joint position distance divided by target height, "
            "then multiplied by 1,000 for display; virtual joints excluded. Each clip "
            "and each source-target pair receive equal weight."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
