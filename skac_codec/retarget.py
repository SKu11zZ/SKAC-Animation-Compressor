from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .math3d import (
    matrix_to_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_matrix,
)
from .metrics import global_joint_positions
from .model import MotionClip, Skeleton


PROFILE_SCHEMA = "skac.retarget_profile"
PROFILE_VERSION = "2.0.0"
LEGACY_PROFILE_VERSION = "1.0.0"
RUNTIME_MODE = "compiled_quaternion_frame_v2"

_CORE_SEMANTICS = {
    "pelvis",
    "spine",
    "neck",
    "head",
    "left_clavicle",
    "right_clavicle",
    "left_upper_arm",
    "right_upper_arm",
    "left_lower_arm",
    "right_lower_arm",
    "left_hand",
    "right_hand",
    "left_thigh",
    "right_thigh",
    "left_calf",
    "right_calf",
    "left_foot",
    "right_foot",
    "left_toe",
    "right_toe",
}


def _plain_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.rsplit(":", 1)[-1].casefold())


_SEMANTIC_ALIASES = {
    "hips": "pelvis",
    "pelvis": "pelvis",
    "root": "root",
    "rootmotion": "root",
    "neck": "neck",
    "neck01": "neck",
    "head": "head",
    "head1": "head",
    "leftshoulder": "left_clavicle",
    "claviclel": "left_clavicle",
    "leftcollar": "left_clavicle",
    "rightshoulder": "right_clavicle",
    "clavicler": "right_clavicle",
    "rightcollar": "right_clavicle",
    "leftarm": "left_upper_arm",
    "upperarml": "left_upper_arm",
    "leftupperarm": "left_upper_arm",
    "leftshouldersmpl": "left_upper_arm",
    "rightarm": "right_upper_arm",
    "upperarmr": "right_upper_arm",
    "rightupperarm": "right_upper_arm",
    "rightshouldersmpl": "right_upper_arm",
    "leftforearm": "left_lower_arm",
    "lowerarml": "left_lower_arm",
    "leftelbow": "left_lower_arm",
    "rightforearm": "right_lower_arm",
    "lowerarmr": "right_lower_arm",
    "rightelbow": "right_lower_arm",
    "lefthand": "left_hand",
    "handl": "left_hand",
    "leftwrist": "left_hand",
    "righthand": "right_hand",
    "handr": "right_hand",
    "rightwrist": "right_hand",
    "leftupleg": "left_thigh",
    "thighl": "left_thigh",
    "lefthip": "left_thigh",
    "rightupleg": "right_thigh",
    "thighr": "right_thigh",
    "righthip": "right_thigh",
    "leftleg": "left_calf",
    "calfl": "left_calf",
    "leftknee": "left_calf",
    "rightleg": "right_calf",
    "calfr": "right_calf",
    "rightknee": "right_calf",
    "leftfoot": "left_foot",
    "footl": "left_foot",
    "leftankle": "left_foot",
    "rightfoot": "right_foot",
    "footr": "right_foot",
    "rightankle": "right_foot",
    "lefttoebase": "left_toe",
    "balll": "left_toe",
    "lefttoe": "left_toe",
    "righttoebase": "right_toe",
    "ballr": "right_toe",
    "righttoe": "right_toe",
}


def semantic_joint_name(name: str) -> str:
    plain = _plain_name(name)
    if plain.startswith("mixamorig"):
        plain = plain[len("mixamorig") :]
    if plain.startswith("spine"):
        return "spine"
    if "twist" in plain:
        return f"twist:{plain}"
    return _SEMANTIC_ALIASES.get(plain, plain)


def _skeleton_semantics(skeleton: Skeleton) -> tuple[str, ...]:
    plain_names = {_plain_name(name) for name in skeleton.names}
    smpl_style = "leftcollar" in plain_names or "rightcollar" in plain_names
    result: list[str] = []
    for name in skeleton.names:
        plain = _plain_name(name)
        if smpl_style and plain == "leftshoulder":
            result.append("left_upper_arm")
        elif smpl_style and plain == "rightshoulder":
            result.append("right_upper_arm")
        else:
            result.append(semantic_joint_name(name))
    return tuple(result)


def _rest_positions(skeleton: Skeleton) -> np.ndarray:
    positions = np.empty((skeleton.joint_count, 3), dtype=np.float64)
    for joint, parent in enumerate(skeleton.parents):
        positions[joint] = skeleton.offsets[joint]
        if parent >= 0:
            positions[joint] += positions[int(parent)]
    return positions


def _children(skeleton: Skeleton) -> list[list[int]]:
    result: list[list[int]] = [[] for _ in range(skeleton.joint_count)]
    for child, parent in enumerate(skeleton.parents):
        if parent >= 0:
            result[int(parent)].append(child)
    return result


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-8 else None


def _joint_basis(skeleton: Skeleton, joint: int, up_axis: int) -> np.ndarray:
    positions = _rest_positions(skeleton)
    child_table = _children(skeleton)
    outgoing = [
        positions[child] - positions[joint]
        for child in child_table[joint]
        if np.linalg.norm(positions[child] - positions[joint]) > 1e-8
    ]
    incoming = None
    parent = int(skeleton.parents[joint])
    if parent >= 0:
        incoming = positions[joint] - positions[parent]

    primary = None
    if outgoing:
        if semantic_joint_name(skeleton.names[joint]) in {"pelvis", "root"}:
            primary = max(outgoing, key=lambda item: abs(float(item[up_axis])))
        else:
            primary = outgoing[0]
    elif incoming is not None:
        primary = incoming
    if primary is None:
        primary = np.eye(3, dtype=np.float64)[up_axis]
    x_axis = _normalize(primary)
    assert x_axis is not None

    hints: list[np.ndarray] = []
    if incoming is not None:
        hints.append(incoming)
    hints.extend(outgoing[1:])
    hints.extend(np.eye(3, dtype=np.float64))
    z_axis = None
    for hint in hints:
        candidate = _normalize(np.cross(x_axis, hint))
        if candidate is not None:
            z_axis = candidate
            break
    if z_axis is None:
        raise ValueError(f"cannot construct a rest basis for joint {skeleton.names[joint]!r}")
    y_axis = _normalize(np.cross(z_axis, x_axis))
    assert y_axis is not None
    return np.column_stack((x_axis, y_axis, z_axis))


def _skeleton_height(skeleton: Skeleton, up_axis: int) -> float:
    semantic_indices = {
        semantic_joint_name(name): index for index, name in enumerate(skeleton.names)
    }

    def chain_length(joint: int) -> float:
        result = 0.0
        while joint != 0:
            result += float(np.linalg.norm(skeleton.offsets[joint]))
            joint = int(skeleton.parents[joint])
        return result

    if "head" in semantic_indices:
        feet = [
            semantic_indices[name]
            for name in ("left_foot", "right_foot")
            if name in semantic_indices
        ]
        if feet:
            humanoid_height = chain_length(semantic_indices["head"]) + float(
                np.mean([chain_length(joint) for joint in feet])
            )
            if humanoid_height > 1e-8:
                return humanoid_height

    positions = _rest_positions(skeleton)
    values = list(positions[:, up_axis])
    for joint, has_end in enumerate(skeleton.has_end_sites):
        if has_end:
            values.append(positions[joint, up_axis] + skeleton.end_site_offsets[joint, up_axis])
    height = float(max(values) - min(values))
    if height <= 1e-8:
        height = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    if height <= 1e-8:
        raise ValueError("skeleton has no usable scale")
    return height


def _spine_indices(skeleton: Skeleton) -> list[int]:
    return [
        index
        for index, semantic in enumerate(_skeleton_semantics(skeleton))
        if semantic == "spine"
    ]


def _semantic_mapping(source: Skeleton, target: Skeleton) -> tuple[dict[int, int], dict[int, str]]:
    source_plain = {_plain_name(name): index for index, name in enumerate(source.names)}
    source_semantics: dict[str, list[int]] = {}
    source_semantic_names = _skeleton_semantics(source)
    target_semantic_names = _skeleton_semantics(target)
    for index, semantic in enumerate(source_semantic_names):
        source_semantics.setdefault(semantic, []).append(index)

    mapping: dict[int, int] = {}
    methods: dict[int, str] = {}
    for target_index, target_name in enumerate(target.names):
        exact = source_plain.get(_plain_name(target_name))
        if exact is not None and source_semantic_names[exact] == target_semantic_names[target_index]:
            mapping[target_index] = exact
            methods[target_index] = "exact_name"
            continue
        semantic = target_semantic_names[target_index]
        matches = source_semantics.get(semantic, [])
        if len(matches) == 1 and semantic != "spine" and not semantic.startswith("twist:"):
            mapping[target_index] = matches[0]
            methods[target_index] = "semantic_alias"

    source_spine = _spine_indices(source)
    target_spine = _spine_indices(target)
    if source_spine and target_spine:
        for position, target_index in enumerate(target_spine):
            source_position = round(
                position * (len(source_spine) - 1) / max(1, len(target_spine) - 1)
            )
            mapping[target_index] = source_spine[source_position]
            methods[target_index] = "normalized_chain"

    source_root_semantic = source_semantic_names[0]
    target_root_semantic = target_semantic_names[0]
    if source_root_semantic != target_root_semantic and target_root_semantic == "root":
        mapping.pop(0, None)
        methods.pop(0, None)
    elif 0 not in mapping and source_root_semantic == target_root_semantic:
        mapping[0] = 0
        methods[0] = "root_semantic"
    return mapping, methods


def _ancestor_evaluation_order(parents: np.ndarray, selected: list[int]) -> tuple[int, ...]:
    required: set[int] = set()
    for joint in selected:
        while joint >= 0 and joint not in required:
            required.add(joint)
            joint = int(parents[joint])
    return tuple(joint for joint in range(len(parents)) if joint in required)


@dataclass(frozen=True)
class JointTransfer:
    target_joint: int
    source_joint: int
    basis_change: np.ndarray
    target_name: str
    source_name: str
    basis_quaternion: np.ndarray | None = None
    mapping_method: str = "legacy"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.basis_change, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("basis_change must be a finite 3x3 matrix")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
            raise ValueError("basis_change must be orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
            raise ValueError("basis_change must be a proper rotation")
        object.__setattr__(self, "basis_change", matrix.copy())
        self.basis_change.setflags(write=False)
        quaternion = (
            matrix_to_quaternion(matrix)
            if self.basis_quaternion is None
            else np.asarray(self.basis_quaternion, dtype=np.float64)
        )
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("basis_quaternion must contain four finite values")
        quaternion = quaternion / np.linalg.norm(quaternion)
        object.__setattr__(self, "basis_quaternion", quaternion.copy())
        self.basis_quaternion.setflags(write=False)
        if not self.mapping_method:
            raise ValueError("mapping_method cannot be empty")


@dataclass(frozen=True)
class RetargetProfile:
    source_skeleton_sha256: str
    target_skeleton_sha256: str
    transfers: tuple[JointTransfer, ...]
    root_translation_scale: float
    up_axis: int = 1
    contact_lock: bool = False
    foot_pairs: tuple[tuple[int, int], ...] = ()
    configuration_scope: str = "source_target_skeleton_pair"
    per_animation_adjustment: bool = False
    schema_version: str = PROFILE_VERSION
    source_evaluation_order: tuple[int, ...] = ()
    target_evaluation_order: tuple[int, ...] = ()
    target_parent_indices: tuple[int, ...] = ()
    shared_core_joint_count: int = 0
    mapped_core_joint_count: int = 0

    def __post_init__(self) -> None:
        if len(self.source_skeleton_sha256) != 64 or len(self.target_skeleton_sha256) != 64:
            raise ValueError("profile requires SHA-256 skeleton signatures")
        if not np.isfinite(self.root_translation_scale) or self.root_translation_scale <= 0:
            raise ValueError("root_translation_scale must be positive and finite")
        if self.up_axis not in {0, 1, 2}:
            raise ValueError("up_axis must be 0, 1, or 2")
        if self.per_animation_adjustment:
            raise ValueError("automatic profiles cannot be animation-specific")
        if self.schema_version not in {LEGACY_PROFILE_VERSION, PROFILE_VERSION}:
            raise ValueError("unsupported retarget profile schema")
        target_indices = [item.target_joint for item in self.transfers]
        if len(target_indices) != len(set(target_indices)):
            raise ValueError("each target joint may appear only once")
        if self.mapped_core_joint_count > self.shared_core_joint_count:
            raise ValueError("mapped core count cannot exceed shared core count")
        if self.schema_version == PROFILE_VERSION:
            if not self.source_evaluation_order or not self.target_evaluation_order:
                raise ValueError("profile v2 requires compiled evaluation orders")
            if not self.target_parent_indices:
                raise ValueError("profile v2 requires target parent indices")

    @property
    def core_coverage(self) -> float:
        if self.shared_core_joint_count == 0:
            return 0.0
        return self.mapped_core_joint_count / self.shared_core_joint_count

    def runtime_plan_dict(self) -> dict[str, Any]:
        return {
            "mode": RUNTIME_MODE,
            "source_joint_indices": [item.source_joint for item in self.transfers],
            "target_joint_indices": [item.target_joint for item in self.transfers],
            "basis_quaternions": [item.basis_quaternion.tolist() for item in self.transfers],
            "source_evaluation_order": list(self.source_evaluation_order),
            "target_evaluation_order": list(self.target_evaluation_order),
            "target_parent_indices": list(self.target_parent_indices),
        }

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "schema_version": self.schema_version,
            "source_skeleton_sha256": self.source_skeleton_sha256,
            "target_skeleton_sha256": self.target_skeleton_sha256,
            "root_translation_scale": self.root_translation_scale,
            "up_axis": self.up_axis,
            "contact_lock": self.contact_lock,
            "foot_pairs": [list(item) for item in self.foot_pairs],
            "configuration_scope": self.configuration_scope,
            "per_animation_adjustment": self.per_animation_adjustment,
            "transfers": [],
        }
        for item in self.transfers:
            transfer_value: dict[str, Any] = {
                    "target_joint": item.target_joint,
                    "source_joint": item.source_joint,
                    "target_name": item.target_name,
                    "source_name": item.source_name,
                    "basis_change": item.basis_change.tolist(),
            }
            if self.schema_version == PROFILE_VERSION:
                transfer_value["basis_quaternion"] = item.basis_quaternion.tolist()
                transfer_value["mapping_method"] = item.mapping_method
            value["transfers"].append(transfer_value)
        if self.schema_version == PROFILE_VERSION:
            value["builder"] = {
                "strategy": "codec_runtime",
                "shared_core_joint_count": self.shared_core_joint_count,
                "mapped_core_joint_count": self.mapped_core_joint_count,
                "core_coverage": self.core_coverage,
            }
            value["runtime_plan"] = self.runtime_plan_dict()
        if include_hash:
            value["profile_sha256"] = self.signature()
        return value

    def signature(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetargetProfile":
        version = value.get("schema_version")
        if value.get("schema") != PROFILE_SCHEMA or version not in {
            LEGACY_PROFILE_VERSION,
            PROFILE_VERSION,
        }:
            raise ValueError("unsupported retarget profile schema")
        runtime = value.get("runtime_plan", {}) if version == PROFILE_VERSION else {}
        builder = value.get("builder", {}) if version == PROFILE_VERSION else {}
        profile = cls(
            source_skeleton_sha256=str(value["source_skeleton_sha256"]),
            target_skeleton_sha256=str(value["target_skeleton_sha256"]),
            transfers=tuple(
                JointTransfer(
                    target_joint=int(item["target_joint"]),
                    source_joint=int(item["source_joint"]),
                    target_name=str(item["target_name"]),
                    source_name=str(item["source_name"]),
                    basis_change=np.asarray(item["basis_change"], dtype=np.float64),
                    basis_quaternion=(
                        np.asarray(item["basis_quaternion"], dtype=np.float64)
                        if "basis_quaternion" in item
                        else None
                    ),
                    mapping_method=str(item.get("mapping_method", "legacy")),
                )
                for item in value["transfers"]
            ),
            root_translation_scale=float(value["root_translation_scale"]),
            up_axis=int(value["up_axis"]),
            contact_lock=bool(value["contact_lock"]),
            foot_pairs=tuple((int(item[0]), int(item[1])) for item in value["foot_pairs"]),
            configuration_scope=str(value["configuration_scope"]),
            per_animation_adjustment=bool(value["per_animation_adjustment"]),
            schema_version=str(version),
            source_evaluation_order=tuple(
                int(item) for item in runtime.get("source_evaluation_order", [])
            ),
            target_evaluation_order=tuple(
                int(item) for item in runtime.get("target_evaluation_order", [])
            ),
            target_parent_indices=tuple(
                int(item) for item in runtime.get("target_parent_indices", [])
            ),
            shared_core_joint_count=int(builder.get("shared_core_joint_count", 0)),
            mapped_core_joint_count=int(builder.get("mapped_core_joint_count", 0)),
        )
        if value.get("profile_sha256") != profile.signature():
            raise ValueError("retarget profile hash does not match")
        if version == PROFILE_VERSION and runtime != profile.runtime_plan_dict():
            raise ValueError("retarget runtime plan does not match its transfers")
        return profile


def build_retarget_profile(
    source: Skeleton,
    target: Skeleton,
    *,
    up_axis: int = 1,
    contact_lock: bool = False,
) -> RetargetProfile:
    if contact_lock:
        raise ValueError("profile v2 does not allow the non-real-time contact correction")
    mapping, methods = _semantic_mapping(source, target)
    transfers: list[JointTransfer] = []
    for target_joint, source_joint in sorted(mapping.items()):
        if _plain_name(source.names[source_joint]) == _plain_name(target.names[target_joint]):
            basis_change = np.eye(3, dtype=np.float64)
        else:
            source_basis = _joint_basis(source, source_joint, up_axis)
            target_basis = _joint_basis(target, target_joint, up_axis)
            basis_change = target_basis @ source_basis.T
        transfers.append(
            JointTransfer(
                target_joint=target_joint,
                source_joint=source_joint,
                basis_change=basis_change,
                target_name=target.names[target_joint],
                source_name=source.names[source_joint],
                basis_quaternion=matrix_to_quaternion(basis_change),
                mapping_method=methods[target_joint],
            )
        )
    if not transfers:
        raise ValueError("source and target skeletons have no semantic joint matches")

    source_semantics = _skeleton_semantics(source)
    target_semantics = _skeleton_semantics(target)
    source_by_semantic = {
        semantic: index for index, semantic in enumerate(source_semantics)
    }
    target_by_semantic = {
        semantic: index for index, semantic in enumerate(target_semantics)
    }
    foot_pairs = tuple(
        (source_by_semantic[semantic], target_by_semantic[semantic])
        for semantic in ("left_foot", "right_foot")
        if semantic in source_by_semantic and semantic in target_by_semantic
    )
    source_core = set(source_semantics) & _CORE_SEMANTICS
    target_core = set(target_semantics) & _CORE_SEMANTICS
    shared_core = source_core & target_core
    mapped_core = {
        target_semantics[target_joint]
        for target_joint, source_joint in mapping.items()
        if target_semantics[target_joint] in shared_core
        and source_semantics[source_joint] == target_semantics[target_joint]
    }
    return RetargetProfile(
        source_skeleton_sha256=source.signature(),
        target_skeleton_sha256=target.signature(),
        transfers=tuple(transfers),
        root_translation_scale=(
            _skeleton_height(target, up_axis) / _skeleton_height(source, up_axis)
        ),
        up_axis=up_axis,
        contact_lock=contact_lock,
        foot_pairs=foot_pairs,
        source_evaluation_order=_ancestor_evaluation_order(
            source.parents, list(mapping.values())
        ),
        target_evaluation_order=_ancestor_evaluation_order(
            target.parents, list(mapping.keys())
        ),
        target_parent_indices=tuple(int(item) for item in target.parents),
        shared_core_joint_count=len(shared_core),
        mapped_core_joint_count=len(mapped_core),
    )


def save_retarget_profile(path: Path, profile: RetargetProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_retarget_profile(path: Path) -> RetargetProfile:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError("retarget profile root must be an object")
    return RetargetProfile.from_dict(value)


def _global_rotation_matrices(clip: MotionClip) -> np.ndarray:
    local = quaternion_to_matrix(clip.local_rotations)
    result = np.empty_like(local)
    for joint, parent in enumerate(clip.skeleton.parents):
        if parent < 0:
            result[:, joint] = local[:, joint]
        else:
            result[:, joint] = result[:, int(parent)] @ local[:, joint]
    return result


def _contact_mask(
    positions: np.ndarray,
    joint: int,
    frame_time: float,
    skeleton_height: float,
    up_axis: int,
) -> np.ndarray:
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    velocity = np.zeros(len(positions), dtype=np.float64)
    if len(positions) > 1:
        velocity[1:] = np.linalg.norm(
            np.diff(positions[:, joint, horizontal_axes], axis=0), axis=1
        ) / frame_time
        velocity[0] = velocity[1]
    height = positions[:, joint, up_axis]
    floor = float(np.min(height))
    return (height <= floor + 0.03 * skeleton_height) & (
        velocity <= 0.15 * skeleton_height
    )


def _apply_contact_lock(
    source: MotionClip,
    target: MotionClip,
    profile: RetargetProfile,
) -> tuple[MotionClip, dict[str, float | int]]:
    if not profile.contact_lock or not profile.foot_pairs:
        return target, {"contact_frames": 0, "root_correction_max": 0.0}
    source_positions = global_joint_positions(source)
    target_positions = global_joint_positions(target)
    source_height = _skeleton_height(source.skeleton, profile.up_axis)
    masks = [
        _contact_mask(
            source_positions,
            source_joint,
            source.frame_time,
            source_height,
            profile.up_axis,
        )
        for source_joint, _ in profile.foot_pairs
    ]
    anchors: list[np.ndarray | None] = [None] * len(profile.foot_pairs)
    previous_active = [False] * len(profile.foot_pairs)
    correction = np.zeros(3, dtype=np.float64)
    corrections = np.zeros((source.frame_count, 3), dtype=np.float64)
    contact_frames = 0
    for frame in range(source.frame_count):
        desired: list[np.ndarray] = []
        for pair_index, ((_, target_joint), mask) in enumerate(
            zip(profile.foot_pairs, masks, strict=True)
        ):
            active = bool(mask[frame])
            raw_position = target_positions[frame, target_joint]
            if active and not previous_active[pair_index]:
                anchors[pair_index] = raw_position + correction
            if active:
                assert anchors[pair_index] is not None
                desired.append(anchors[pair_index] - raw_position)
                contact_frames += 1
            else:
                anchors[pair_index] = None
            previous_active[pair_index] = active
        if desired:
            correction = np.mean(desired, axis=0)
        else:
            correction *= 0.8
        corrections[frame] = correction

    translations = target.local_translations.copy()
    translations[:, 0] += corrections
    locked = MotionClip(
        skeleton=target.skeleton,
        local_rotations=target.local_rotations,
        local_translations=translations,
        frame_time=target.frame_time,
    )
    return locked, {
        "contact_frames": contact_frames,
        "root_correction_max": float(np.max(np.linalg.norm(corrections, axis=1))),
    }


def retarget_motion_reference(
    source: MotionClip,
    target_skeleton: Skeleton,
    profile: RetargetProfile,
) -> tuple[MotionClip, dict[str, Any]]:
    if source.skeleton.signature() != profile.source_skeleton_sha256:
        raise ValueError("source skeleton does not match the frozen profile")
    if target_skeleton.signature() != profile.target_skeleton_sha256:
        raise ValueError("target skeleton does not match the frozen profile")
    for transfer in profile.transfers:
        if not 0 <= transfer.source_joint < source.skeleton.joint_count:
            raise ValueError("profile source joint is outside the source skeleton")
        if not 0 <= transfer.target_joint < target_skeleton.joint_count:
            raise ValueError("profile target joint is outside the target skeleton")
        if source.skeleton.names[transfer.source_joint] != transfer.source_name:
            raise ValueError("profile source joint name does not match the source skeleton")
        if target_skeleton.names[transfer.target_joint] != transfer.target_name:
            raise ValueError("profile target joint name does not match the target skeleton")
    for source_joint, target_joint in profile.foot_pairs:
        if not 0 <= source_joint < source.skeleton.joint_count:
            raise ValueError("profile foot source is outside the source skeleton")
        if not 0 <= target_joint < target_skeleton.joint_count:
            raise ValueError("profile foot target is outside the target skeleton")
    if not all(
        channel in target_skeleton.channels[0]
        for channel in ("Xposition", "Yposition", "Zposition")
    ):
        raise ValueError("target root requires X/Y/Z position channels")

    source_global = _global_rotation_matrices(source)
    target_local = np.broadcast_to(
        np.eye(3), (source.frame_count, target_skeleton.joint_count, 3, 3)
    ).copy()
    target_global = np.empty_like(target_local)
    transfers = {item.target_joint: item for item in profile.transfers}
    for target_joint, parent in enumerate(target_skeleton.parents):
        transfer = transfers.get(target_joint)
        if transfer is not None:
            change = transfer.basis_change
            desired_global = (
                change
                @ source_global[:, transfer.source_joint]
                @ change.T
            )
            if parent < 0:
                target_local[:, target_joint] = desired_global
            else:
                target_local[:, target_joint] = (
                    np.swapaxes(target_global[:, int(parent)], -1, -2)
                    @ desired_global
                )
        if parent < 0:
            target_global[:, target_joint] = target_local[:, target_joint]
        else:
            target_global[:, target_joint] = (
                target_global[:, int(parent)] @ target_local[:, target_joint]
            )

    rotations = matrix_to_quaternion(target_local)
    translations = np.broadcast_to(
        target_skeleton.offsets,
        (source.frame_count, target_skeleton.joint_count, 3),
    ).copy()
    source_root_motion = source.local_translations[:, 0] - source.skeleton.offsets[0]
    translations[:, 0] += source_root_motion * profile.root_translation_scale
    target = MotionClip(
        skeleton=target_skeleton,
        local_rotations=rotations,
        local_translations=translations,
        frame_time=source.frame_time,
    )
    target, contact_diagnostics = _apply_contact_lock(source, target, profile)
    return target, {
        "profile_sha256": profile.signature(),
        "profile_schema_version": profile.schema_version,
        "runtime_mode": "matrix_batch_reference",
        "mapped_joint_count": len(profile.transfers),
        "target_joint_count": target_skeleton.joint_count,
        "root_translation_scale": profile.root_translation_scale,
        "core_coverage": profile.core_coverage,
        **contact_diagnostics,
    }


def _quaternion_multiply_into(left: np.ndarray, right: np.ndarray, output: np.ndarray) -> None:
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    output[0] = aw * bw - ax * bx - ay * by - az * bz
    output[1] = aw * bx + ax * bw + ay * bz - az * by
    output[2] = aw * by - ax * bz + ay * bw + az * bx
    output[3] = aw * bz + ax * by - ay * bx + az * bw


def _quaternion_conjugate_into(value: np.ndarray, output: np.ndarray) -> None:
    output[0] = value[0]
    output[1] = -value[1]
    output[2] = -value[2]
    output[3] = -value[3]


@dataclass
class RetargetRuntime:
    """A compiled, reusable, allocation-free-per-frame retarget plan."""

    source_skeleton: Skeleton
    target_skeleton: Skeleton
    profile: RetargetProfile
    source_evaluation_order: tuple[int, ...]
    target_evaluation_order: tuple[int, ...]
    source_indices: np.ndarray
    transfer_by_target: np.ndarray
    basis_quaternions: np.ndarray
    basis_conjugates: np.ndarray
    basis_identity: np.ndarray
    source_global: np.ndarray
    target_global: np.ndarray
    temporary_a: np.ndarray
    temporary_b: np.ndarray

    def evaluate_frame_into(
        self,
        source_local_rotations: np.ndarray,
        source_root_translation: np.ndarray,
        output_local_rotations: np.ndarray,
        output_local_translations: np.ndarray,
    ) -> None:
        source_local_rotations = np.asarray(source_local_rotations, dtype=np.float64)
        source_root_translation = np.asarray(source_root_translation, dtype=np.float64)
        if source_local_rotations.shape != (self.source_skeleton.joint_count, 4):
            raise ValueError("source frame rotations have the wrong shape")
        if source_root_translation.shape != (3,):
            raise ValueError("source root translation must contain three values")
        if output_local_rotations.shape != (self.target_skeleton.joint_count, 4):
            raise ValueError("target rotation output has the wrong shape")
        if output_local_translations.shape != (self.target_skeleton.joint_count, 3):
            raise ValueError("target translation output has the wrong shape")

        output_local_rotations.fill(0.0)
        output_local_rotations[:, 0] = 1.0
        np.copyto(output_local_translations, self.target_skeleton.offsets)

        for joint in self.source_evaluation_order:
            parent = int(self.source_skeleton.parents[joint])
            if parent < 0:
                np.copyto(self.source_global[joint], source_local_rotations[joint])
            else:
                _quaternion_multiply_into(
                    self.source_global[parent],
                    source_local_rotations[joint],
                    self.source_global[joint],
                )

        for target_joint in self.target_evaluation_order:
            parent = int(self.target_skeleton.parents[target_joint])
            transfer_slot = int(self.transfer_by_target[target_joint])
            if transfer_slot >= 0:
                source_joint = int(self.source_indices[transfer_slot])
                if self.basis_identity[transfer_slot]:
                    desired_global = self.source_global[source_joint]
                else:
                    _quaternion_multiply_into(
                        self.basis_quaternions[transfer_slot],
                        self.source_global[source_joint],
                        self.temporary_a,
                    )
                    _quaternion_multiply_into(
                        self.temporary_a,
                        self.basis_conjugates[transfer_slot],
                        self.temporary_b,
                    )
                    desired_global = self.temporary_b
                if parent < 0:
                    np.copyto(output_local_rotations[target_joint], desired_global)
                else:
                    _quaternion_conjugate_into(
                        self.target_global[parent], self.temporary_a
                    )
                    _quaternion_multiply_into(
                        self.temporary_a,
                        desired_global,
                        output_local_rotations[target_joint],
                    )
            if parent < 0:
                np.copyto(self.target_global[target_joint], output_local_rotations[target_joint])
            else:
                _quaternion_multiply_into(
                    self.target_global[parent],
                    output_local_rotations[target_joint],
                    self.target_global[target_joint],
                )

        for axis in range(3):
            output_local_translations[0, axis] += (
                float(source_root_translation[axis])
                - float(self.source_skeleton.offsets[0, axis])
            ) * self.profile.root_translation_scale


def compile_retarget_profile(
    source_skeleton: Skeleton,
    target_skeleton: Skeleton,
    profile: RetargetProfile,
) -> RetargetRuntime:
    if source_skeleton.signature() != profile.source_skeleton_sha256:
        raise ValueError("source skeleton does not match the frozen profile")
    if target_skeleton.signature() != profile.target_skeleton_sha256:
        raise ValueError("target skeleton does not match the frozen profile")
    if profile.contact_lock:
        raise ValueError("contact lock is not supported by the real-time runtime plan")
    if not all(
        channel in target_skeleton.channels[0]
        for channel in ("Xposition", "Yposition", "Zposition")
    ):
        raise ValueError("target root requires X/Y/Z position channels")

    for transfer in profile.transfers:
        if not 0 <= transfer.source_joint < source_skeleton.joint_count:
            raise ValueError("profile source joint is outside the source skeleton")
        if not 0 <= transfer.target_joint < target_skeleton.joint_count:
            raise ValueError("profile target joint is outside the target skeleton")
        if source_skeleton.names[transfer.source_joint] != transfer.source_name:
            raise ValueError("profile source joint name does not match the source skeleton")
        if target_skeleton.names[transfer.target_joint] != transfer.target_name:
            raise ValueError("profile target joint name does not match the target skeleton")

    source_order = _ancestor_evaluation_order(
        source_skeleton.parents, [item.source_joint for item in profile.transfers]
    )
    target_order = _ancestor_evaluation_order(
        target_skeleton.parents, [item.target_joint for item in profile.transfers]
    )
    if profile.schema_version == PROFILE_VERSION:
        if source_order != profile.source_evaluation_order:
            raise ValueError("profile source evaluation order is invalid")
        if target_order != profile.target_evaluation_order:
            raise ValueError("profile target evaluation order is invalid")
        if tuple(int(item) for item in target_skeleton.parents) != profile.target_parent_indices:
            raise ValueError("profile target parent table is invalid")

    source_indices = np.asarray(
        [item.source_joint for item in profile.transfers], dtype=np.int32
    )
    transfer_by_target = np.full(target_skeleton.joint_count, -1, dtype=np.int32)
    basis_quaternions = np.asarray(
        [item.basis_quaternion for item in profile.transfers], dtype=np.float64
    )
    for slot, item in enumerate(profile.transfers):
        transfer_by_target[item.target_joint] = slot
    basis_conjugates = basis_quaternions.copy()
    basis_conjugates[:, 1:] *= -1.0
    basis_identity = (
        np.abs(basis_quaternions[:, 0]) >= 1.0 - 1e-12
    ) & (np.max(np.abs(basis_quaternions[:, 1:]), axis=1) <= 1e-12)
    for value in (
        source_indices,
        transfer_by_target,
        basis_quaternions,
        basis_conjugates,
        basis_identity,
    ):
        value.setflags(write=False)
    return RetargetRuntime(
        source_skeleton=source_skeleton,
        target_skeleton=target_skeleton,
        profile=profile,
        source_evaluation_order=source_order,
        target_evaluation_order=target_order,
        source_indices=source_indices,
        transfer_by_target=transfer_by_target,
        basis_quaternions=basis_quaternions,
        basis_conjugates=basis_conjugates,
        basis_identity=basis_identity,
        source_global=np.empty((source_skeleton.joint_count, 4), dtype=np.float64),
        target_global=np.empty((target_skeleton.joint_count, 4), dtype=np.float64),
        temporary_a=np.empty(4, dtype=np.float64),
        temporary_b=np.empty(4, dtype=np.float64),
    )


def retarget_motion(
    source: MotionClip,
    target_skeleton: Skeleton,
    profile: RetargetProfile,
    *,
    runtime: RetargetRuntime | None = None,
) -> tuple[MotionClip, dict[str, Any]]:
    runtime = runtime or compile_retarget_profile(source.skeleton, target_skeleton, profile)
    if runtime.source_skeleton.signature() != source.skeleton.signature():
        raise ValueError("compiled runtime uses a different source skeleton")
    if runtime.target_skeleton.signature() != target_skeleton.signature():
        raise ValueError("compiled runtime uses a different target skeleton")

    source_global = np.empty(
        (source.frame_count, source.skeleton.joint_count, 4), dtype=np.float64
    )
    for joint in runtime.source_evaluation_order:
        parent = int(source.skeleton.parents[joint])
        if parent < 0:
            source_global[:, joint] = source.local_rotations[:, joint]
        else:
            source_global[:, joint] = quaternion_multiply(
                source_global[:, parent], source.local_rotations[:, joint]
            )

    rotations = np.zeros(
        (source.frame_count, target_skeleton.joint_count, 4), dtype=np.float64
    )
    rotations[..., 0] = 1.0
    target_global = np.empty_like(rotations)
    for target_joint in runtime.target_evaluation_order:
        parent = int(target_skeleton.parents[target_joint])
        transfer_slot = int(runtime.transfer_by_target[target_joint])
        if transfer_slot >= 0:
            source_joint = int(runtime.source_indices[transfer_slot])
            if runtime.basis_identity[transfer_slot]:
                desired_global = source_global[:, source_joint]
            else:
                desired_global = quaternion_multiply(
                    runtime.basis_quaternions[transfer_slot],
                    source_global[:, source_joint],
                )
                desired_global = quaternion_multiply(
                    desired_global, runtime.basis_conjugates[transfer_slot]
                )
            if parent < 0:
                rotations[:, target_joint] = desired_global
            else:
                rotations[:, target_joint] = quaternion_multiply(
                    quaternion_conjugate(target_global[:, parent]), desired_global
                )
        if parent < 0:
            target_global[:, target_joint] = rotations[:, target_joint]
        else:
            target_global[:, target_joint] = quaternion_multiply(
                target_global[:, parent], rotations[:, target_joint]
            )

    translations = np.broadcast_to(
        target_skeleton.offsets,
        (source.frame_count, target_skeleton.joint_count, 3),
    ).copy()
    source_root_motion = source.local_translations[:, 0] - source.skeleton.offsets[0]
    translations[:, 0] += source_root_motion * profile.root_translation_scale
    target = MotionClip(
        skeleton=target_skeleton,
        local_rotations=rotations,
        local_translations=translations,
        frame_time=source.frame_time,
    )
    return target, {
        "profile_sha256": profile.signature(),
        "profile_schema_version": profile.schema_version,
        "runtime_mode": RUNTIME_MODE,
        "mapped_joint_count": len(profile.transfers),
        "target_joint_count": target_skeleton.joint_count,
        "root_translation_scale": profile.root_translation_scale,
        "core_coverage": profile.core_coverage,
        "source_runtime_joint_count": len(runtime.source_evaluation_order),
        "target_runtime_joint_count": len(runtime.target_evaluation_order),
        "contact_frames": 0,
        "root_correction_max": 0.0,
    }
