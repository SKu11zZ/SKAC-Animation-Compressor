from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


TEST_CHARACTERS = ("Mousey_m", "Goblin_m", "Mremireh_m", "Vampire_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score external predictions with the SAN official-code MSE."
    )
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--upstream-utils-root", type=Path, required=True)
    parser.add_argument("--upstream-retargeting-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method-name", default="public_rotation_copy_pipeline_baseline"
    )
    return parser.parse_args()


def load_upstream_modules(utils_root: Path, retargeting_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(utils_root.resolve()))
    sys.path.insert(0, str(retargeting_root.resolve()))
    import Animation  # type: ignore[import-not-found]
    import BVH  # type: ignore[import-not-found]
    from datasets.bvh_parser import BVH_file  # type: ignore[import-not-found]

    return {"Animation": Animation, "BVH": BVH, "BVH_file": BVH_file}


def score_pair(
    modules: dict[str, Any],
    prediction_dir: Path,
    truth_dir: Path,
    target: str,
    reference_root: Path,
) -> dict[str, float | int]:
    height = float(
        modules["BVH_file"](str(reference_root / f"{target}.bvh")).get_height()
    )
    predictions = sorted(prediction_dir.glob("*.bvh"), key=lambda path: int(path.stem))
    if not predictions:
        raise ValueError(f"no predictions in {prediction_dir}")
    errors: list[float] = []
    for prediction in predictions:
        truth = truth_dir / f"{prediction.stem}_gt.bvh"
        if not truth.is_file():
            raise FileNotFoundError(truth)
        animation, names, _ = modules["BVH"].load(str(prediction))
        reference, _, _ = modules["BVH"].load(str(truth))
        indices = [index for index, name in enumerate(names) if "virtual" not in name]
        positions = modules["Animation"].positions_global(animation)[:, indices, :]
        truth_positions = modules["Animation"].positions_global(reference)[:, indices, :]
        if positions.shape != truth_positions.shape:
            raise ValueError(
                f"shape mismatch for {prediction.name}: "
                f"{positions.shape} != {truth_positions.shape}"
            )
        errors.append(float(np.mean(np.square(positions - truth_positions)) / height**2))
    raw = float(np.mean(errors))
    return {"sample_count": len(errors), "mse_raw": raw, "mse_x1e3": raw * 1_000.0}


def main() -> None:
    args = parse_args()
    modules = load_upstream_modules(
        args.upstream_utils_root, args.upstream_retargeting_root
    )
    cross_details: dict[str, dict[str, float | int]] = {}
    for target in TEST_CHARACTERS:
        cross_details[target] = score_pair(
            modules,
            args.predictions_root / "cross_structure" / target,
            args.ground_truth_root / "cross_structure" / target,
            target,
            args.reference_root,
        )

    intra_details: dict[str, dict[str, dict[str, float | int]]] = {}
    all_intra: list[float] = []
    non_identity_intra: list[float] = []
    for source in TEST_CHARACTERS:
        source_details: dict[str, dict[str, float | int]] = {}
        for target in TEST_CHARACTERS:
            score = score_pair(
                modules,
                args.predictions_root / "intra_structure" / f"from_{source}" / target,
                args.ground_truth_root / "intra_structure" / f"from_{source}" / target,
                target,
                args.reference_root,
            )
            source_details[target] = score
            value = float(score["mse_raw"])
            all_intra.append(value)
            if source != target:
                non_identity_intra.append(value)
        intra_details[f"from_{source}"] = source_details

    cross_raw = float(np.mean([float(item["mse_raw"]) for item in cross_details.values()]))
    intra_raw = float(np.mean(all_intra))
    non_identity_raw = float(np.mean(non_identity_intra))
    report = {
        "report_type": "SAN_official_code_metric",
        "method": args.method_name,
        "cross_structure": {
            "mse_raw": cross_raw,
            "mse_x1e3": cross_raw * 1_000.0,
            "target_details": cross_details,
        },
        "intra_structure": {
            "official_all_pairs_mse_raw": intra_raw,
            "official_all_pairs_mse_x1e3": intra_raw * 1_000.0,
            "non_identity_pairs_mse_raw": non_identity_raw,
            "non_identity_pairs_mse_x1e3": non_identity_raw * 1_000.0,
            "source_target_details": intra_details,
        },
        "metric_definition": (
            "Mean squared global-joint coordinate error divided by squared target "
            "height; virtual joints excluded."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
