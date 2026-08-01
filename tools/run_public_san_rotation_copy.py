from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from skac_public_core import (  # noqa: E402
    Motion as CoreMotion,
    RetargetProfile as CoreRetargetProfile,
    Skeleton as CoreSkeleton,
    build_profile as build_core_profile,
    canonical_joint_name,
    retarget_motion as retarget_core_motion,
)


CROSS_SOURCE = "BigVegas"
TEST_CHARACTERS = ("Mousey_m", "Goblin_m", "Mremireh_m", "Vampire_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic public rotation-copy baseline on the SAN test set. "
            "This validates the benchmark pipeline; it is not a product-method score."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--evaluation-skeleton-root", type=Path, required=True)
    parser.add_argument("--upstream-utils-root", type=Path, required=True)
    parser.add_argument("--upstream-retargeting-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--solver", choices=("rotation_copy", "ccd_ik"), default="rotation_copy"
    )
    parser.add_argument("--ik-iterations", type=int, default=2)
    return parser.parse_args()


def build_index_mapping(
    source_names: list[str], target_names: list[str]
) -> list[tuple[int, int]]:
    source_indices: dict[str, int] = {}
    for index, name in enumerate(source_names):
        source_indices.setdefault(canonical_joint_name(name), index)
    return [
        (target_index, source_indices[canonical_joint_name(target_name)])
        for target_index, target_name in enumerate(target_names)
        if canonical_joint_name(target_name) in source_indices
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenProfile:
    source: str
    target: str
    source_reference: Path
    target_reference: Path
    target_scale_reference: Path
    source_names: list[str]
    target_names: list[str]
    mapping: list[tuple[int, int]]
    translation_scale: float
    core_profile: CoreRetargetProfile

    def public_record(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "profile_type": "public_semantic_joint_mapping",
            "source_character": self.source,
            "target_character": self.target,
            "source_reference_sha256": sha256_file(self.source_reference),
            "target_reference_sha256": sha256_file(self.target_reference),
            "target_scale_reference_sha256": sha256_file(
                self.target_scale_reference
            ),
            "source_joint_count": len(self.source_names),
            "target_joint_count": len(self.target_names),
            "mapped_joint_count": len(self.mapping),
            "translation_scale": self.translation_scale,
            "joint_mapping": [
                {
                    "source": canonical_joint_name(self.source_names[source_index]),
                    "target": canonical_joint_name(self.target_names[target_index]),
                }
                for target_index, source_index in self.mapping
            ],
            "configuration_scope": "source_target_skeleton_pair",
            "per_animation_adjustment": False,
        }


def load_upstream_modules(utils_root: Path, retargeting_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(utils_root.resolve()))
    sys.path.insert(0, str(retargeting_root.resolve()))
    import Animation  # type: ignore[import-not-found]
    import BVH  # type: ignore[import-not-found]
    import BVH_mod  # type: ignore[import-not-found]
    from Quaternions_old import Quaternions  # type: ignore[import-not-found]
    from datasets.bvh_parser import BVH_file  # type: ignore[import-not-found]

    return {
        "Animation": Animation,
        "BVH": BVH,
        "BVH_mod": BVH_mod,
        "BVH_file": BVH_file,
        "Quaternions": Quaternions,
    }


def build_profile(
    modules: dict[str, Any],
    reference_root: Path,
    evaluation_skeleton_root: Path,
    source: str,
    target: str,
) -> FrozenProfile:
    source_reference = reference_root / f"{source}.bvh"
    target_scale_reference = reference_root / f"{target}.bvh"
    target_reference = evaluation_skeleton_root / target / "0_gt.bvh"
    source_reference_animation, source_names, _ = modules["BVH_mod"].load(
        str(source_reference), need_quater=True
    )
    target_reference_animation, target_names, _ = modules["BVH_mod"].load(
        str(target_reference), need_quater=True
    )
    source_height = float(modules["BVH_file"](str(source_reference)).get_height())
    target_height = float(
        modules["BVH_file"](str(target_scale_reference)).get_height()
    )
    if source_height <= 0 or target_height <= 0:
        raise ValueError("reference skeleton height must be positive")
    core_profile = build_core_profile(
        CoreSkeleton(
            names=tuple(source_names),
            parents=source_reference_animation.parents,
            offsets=source_reference_animation.offsets,
            height=source_height,
        ),
        CoreSkeleton(
            names=tuple(target_names),
            parents=target_reference_animation.parents,
            offsets=target_reference_animation.offsets,
            height=target_height,
        ),
    )
    return FrozenProfile(
        source=source,
        target=target,
        source_reference=source_reference,
        target_reference=target_reference,
        target_scale_reference=target_scale_reference,
        source_names=source_names,
        target_names=target_names,
        mapping=list(core_profile.target_to_source),
        translation_scale=core_profile.root_translation_scale,
        core_profile=core_profile,
    )


def retarget_clip(
    modules: dict[str, Any],
    profile: FrozenProfile,
    source_path: Path,
    output: Path,
    solver: str,
    ik_iterations: int,
) -> int:
    source_animation, source_names, frame_time = modules["BVH_mod"].load(
        str(source_path), need_quater=True
    )
    if source_names != profile.source_names:
        raise ValueError(f"source skeleton changed for {source_path.name}")
    source_animation = source_animation[::2]
    frame_count = len(source_animation) // 4 * 4
    source_animation = source_animation[:frame_count]
    if frame_count == 0:
        raise ValueError(f"no complete evaluation frames in {source_path.name}")

    target_reference, target_names, _ = modules["BVH_mod"].load(
        str(profile.target_reference), need_quater=True
    )
    if target_names != profile.target_names:
        raise ValueError("target reference skeleton changed after profile freeze")
    joint_count = len(target_names)
    core_result = retarget_core_motion(
        profile.core_profile,
        CoreMotion(
            local_rotations=source_animation.rotations.qs,
            root_positions=source_animation.positions[:, 0, :],
        ),
    )
    target_animation = modules["Animation"].Animation(
        modules["Quaternions"](core_result.local_rotations.copy()),
        np.repeat(target_reference.offsets[np.newaxis, :, :], frame_count, axis=0),
        target_reference.orients.copy(),
        target_reference.offsets.copy(),
        target_reference.parents.copy(),
    )
    target_animation.positions[:, 0, :] = core_result.root_positions
    if solver == "ccd_ik":
        apply_ccd_ik(
            modules,
            source_animation,
            source_names,
            target_animation,
            target_names,
            profile.translation_scale,
            ik_iterations,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    modules["BVH"].save(
        str(output),
        target_animation,
        names=target_names,
        frametime=frame_time * 2.0,
        order="zyx",
    )
    return frame_count


def apply_ccd_ik(
    modules: dict[str, Any],
    source_animation: Any,
    source_names: list[str],
    target_animation: Any,
    target_names: list[str],
    translation_scale: float,
    iterations: int,
) -> None:
    if iterations <= 0:
        raise ValueError("IK iterations must be positive")
    source_indices = {
        canonical_joint_name(name): index for index, name in enumerate(source_names)
    }
    target_indices = {
        canonical_joint_name(name): index for index, name in enumerate(target_names)
    }
    chain_specs = {
        "LeftHand": ("LeftShoulder", ("LeftForeArm", "LeftArm", "LeftShoulder")),
        "RightHand": (
            "RightShoulder",
            ("RightForeArm", "RightArm", "RightShoulder"),
        ),
        "LeftFoot": ("LeftUpLeg", ("LeftLeg", "LeftUpLeg")),
        "RightFoot": ("RightUpLeg", ("RightLeg", "RightUpLeg")),
        "Head": ("Spine", ("Neck", "Spine2", "Spine1", "Spine")),
    }
    source_positions = modules["Animation"].positions_global(source_animation)
    target_positions = modules["Animation"].positions_global(target_animation)

    def chain_length(animation: Any, end_index: int, root_index: int) -> float:
        length = 0.0
        current = end_index
        while current != root_index:
            if current < 0:
                raise ValueError("chain root is not an ancestor of the end effector")
            length += float(np.linalg.norm(animation.offsets[current]))
            current = int(animation.parents[current])
        return length

    goals: dict[str, np.ndarray] = {}
    for effector, (chain_root, _) in chain_specs.items():
        required_source = (effector, chain_root)
        required_target = (effector, chain_root)
        if any(name not in source_indices for name in required_source) or any(
            name not in target_indices for name in required_target
        ):
            continue
        source_end_index = source_indices[effector]
        source_root_index = source_indices[chain_root]
        target_end_index = target_indices[effector]
        target_root_index = target_indices[chain_root]
        source_length = chain_length(
            source_animation, source_end_index, source_root_index
        )
        target_length = chain_length(
            target_animation, target_end_index, target_root_index
        )
        chain_scale = (
            target_length / source_length
            if source_length > 1e-8
            else translation_scale
        )
        source_vector = (
            source_positions[:, source_end_index, :]
            - source_positions[:, source_root_index, :]
        )
        goals[effector] = (
            target_positions[:, target_root_index, :] + source_vector * chain_scale
        )

    for _ in range(iterations):
        for effector, (_, chain) in chain_specs.items():
            if effector not in goals:
                continue
            effector_index = target_indices[effector]
            for joint_name in chain:
                if joint_name not in target_indices:
                    continue
                joint_index = target_indices[joint_name]
                global_positions = modules["Animation"].positions_global(
                    target_animation
                )
                joint_positions = global_positions[:, joint_index, :]
                current_vectors = (
                    global_positions[:, effector_index, :] - joint_positions
                )
                goal_vectors = goals[effector] - joint_positions
                valid = (
                    np.linalg.norm(current_vectors, axis=-1) > 1e-8
                ) & (np.linalg.norm(goal_vectors, axis=-1) > 1e-8)
                if not np.any(valid):
                    continue
                delta = modules["Quaternions"].id(len(target_animation))
                delta.qs[valid] = modules["Quaternions"].between(
                    current_vectors[valid], goal_vectors[valid]
                ).qs
                global_rotations = modules["Animation"].rotations_global(
                    target_animation
                )
                new_global = delta * global_rotations[:, joint_index]
                parent_index = int(target_animation.parents[joint_index])
                if parent_index < 0:
                    new_local = new_global
                else:
                    new_local = (-global_rotations[:, parent_index]) * new_global
                target_animation.rotations.qs[:, joint_index, :] = new_local.qs


def save_profile(profile: FrozenProfile, profile_root: Path) -> None:
    destination = profile_root / f"{profile.source}__to__{profile.target}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile.public_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.ik_iterations <= 0:
        raise ValueError("--ik-iterations must be positive")
    modules = load_upstream_modules(
        args.upstream_utils_root, args.upstream_retargeting_root
    )
    motions = (args.data_root / "test_list.txt").read_text(encoding="utf-8").splitlines()
    if args.limit is not None:
        motions = motions[: args.limit]
    if not motions:
        raise ValueError("test motion list is empty")

    pairs: list[tuple[str, str, str]] = []
    pairs.extend(("cross_structure", CROSS_SOURCE, target) for target in TEST_CHARACTERS)
    pairs.extend(
        ("intra_structure", source, target)
        for source in TEST_CHARACTERS
        for target in TEST_CHARACTERS
    )

    profiles: dict[tuple[str, str], FrozenProfile] = {}
    for _, source, target in pairs:
        key = (source, target)
        if key not in profiles:
            profile = build_profile(
                modules,
                args.reference_root,
                args.evaluation_skeleton_root,
                source,
                target,
            )
            profiles[key] = profile
            save_profile(profile, args.output_root / "profiles")

    total_frames = 0
    completed = 0
    for group, source, target in pairs:
        profile = profiles[(source, target)]
        if group == "cross_structure":
            destination = args.output_root / group / target
        else:
            destination = args.output_root / group / f"from_{source}" / target
        for index, motion in enumerate(motions):
            total_frames += retarget_clip(
                modules,
                profile,
                args.data_root / source / motion,
                destination / f"{index}.bvh",
                args.solver,
                args.ik_iterations,
            )
            completed += 1

    summary = {
        "method": (
            "public_rotation_copy_pipeline_baseline"
            if args.solver == "rotation_copy"
            else "public_ccd_ik_pipeline_prototype"
        ),
        "solver": args.solver,
        "ik_iterations": args.ik_iterations if args.solver == "ccd_ik" else 0,
        "status": "completed",
        "motion_count": len(motions),
        "profile_count": len(profiles),
        "prediction_count": completed,
        "frame_count": total_frames,
        "per_animation_adjustment": False,
        "claim_scope": "pipeline baseline only; not a product-method result",
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
