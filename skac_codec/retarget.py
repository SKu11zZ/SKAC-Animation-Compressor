from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .math3d import matrix_to_quaternion, quaternion_to_matrix
from .metrics import global_joint_positions
from .model import MotionClip, Skeleton


PROFILE_SCHEMA = "skac.retarget_profile"
PROFILE_VERSION = "1.0.0"


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
        for index, name in enumerate(skeleton.names)
        if semantic_joint_name(name) == "spine"
    ]


def _semantic_mapping(source: Skeleton, target: Skeleton) -> dict[int, int]:
    source_plain = {_plain_name(name): index for index, name in enumerate(source.names)}
    source_semantics: dict[str, list[int]] = {}
    for index, name in enumerate(source.names):
        source_semantics.setdefault(semantic_joint_name(name), []).append(index)

    mapping: dict[int, int] = {}
    for target_index, target_name in enumerate(target.names):
        exact = source_plain.get(_plain_name(target_name))
        if exact is not None:
            mapping[target_index] = exact
            continue
        semantic = semantic_joint_name(target_name)
        matches = source_semantics.get(semantic, [])
        if len(matches) == 1 and semantic != "spine" and not semantic.startswith("twist:"):
            mapping[target_index] = matches[0]

    source_spine = _spine_indices(source)
    target_spine = _spine_indices(target)
    if source_spine and target_spine:
        for position, target_index in enumerate(target_spine):
            source_position = round(
                position * (len(source_spine) - 1) / max(1, len(target_spine) - 1)
            )
            mapping[target_index] = source_spine[source_position]

    source_root_semantic = semantic_joint_name(source.names[0])
    target_root_semantic = semantic_joint_name(target.names[0])
    if source_root_semantic != target_root_semantic and target_root_semantic == "root":
        mapping.pop(0, None)
    elif 0 not in mapping and source_root_semantic == target_root_semantic:
        mapping[0] = 0
    return mapping


@dataclass(frozen=True)
class JointTransfer:
    target_joint: int
    source_joint: int
    basis_change: np.ndarray
    target_name: str
    source_name: str

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

    def __post_init__(self) -> None:
        if len(self.source_skeleton_sha256) != 64 or len(self.target_skeleton_sha256) != 64:
            raise ValueError("profile requires SHA-256 skeleton signatures")
        if not np.isfinite(self.root_translation_scale) or self.root_translation_scale <= 0:
            raise ValueError("root_translation_scale must be positive and finite")
        if self.up_axis not in {0, 1, 2}:
            raise ValueError("up_axis must be 0, 1, or 2")
        if self.per_animation_adjustment:
            raise ValueError("automatic profiles cannot be animation-specific")
        target_indices = [item.target_joint for item in self.transfers]
        if len(target_indices) != len(set(target_indices)):
            raise ValueError("each target joint may appear only once")

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "schema_version": PROFILE_VERSION,
            "source_skeleton_sha256": self.source_skeleton_sha256,
            "target_skeleton_sha256": self.target_skeleton_sha256,
            "root_translation_scale": self.root_translation_scale,
            "up_axis": self.up_axis,
            "contact_lock": self.contact_lock,
            "foot_pairs": [list(item) for item in self.foot_pairs],
            "configuration_scope": self.configuration_scope,
            "per_animation_adjustment": self.per_animation_adjustment,
            "transfers": [
                {
                    "target_joint": item.target_joint,
                    "source_joint": item.source_joint,
                    "target_name": item.target_name,
                    "source_name": item.source_name,
                    "basis_change": item.basis_change.tolist(),
                }
                for item in self.transfers
            ],
        }
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
        if value.get("schema") != PROFILE_SCHEMA or value.get("schema_version") != PROFILE_VERSION:
            raise ValueError("unsupported retarget profile schema")
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
                )
                for item in value["transfers"]
            ),
            root_translation_scale=float(value["root_translation_scale"]),
            up_axis=int(value["up_axis"]),
            contact_lock=bool(value["contact_lock"]),
            foot_pairs=tuple((int(item[0]), int(item[1])) for item in value["foot_pairs"]),
            configuration_scope=str(value["configuration_scope"]),
            per_animation_adjustment=bool(value["per_animation_adjustment"]),
        )
        if value.get("profile_sha256") != profile.signature():
            raise ValueError("retarget profile hash does not match")
        return profile


def build_retarget_profile(
    source: Skeleton,
    target: Skeleton,
    *,
    up_axis: int = 1,
    contact_lock: bool = False,
) -> RetargetProfile:
    mapping = _semantic_mapping(source, target)
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
            )
        )
    if not transfers:
        raise ValueError("source and target skeletons have no semantic joint matches")

    source_by_semantic = {
        semantic_joint_name(name): index for index, name in enumerate(source.names)
    }
    target_by_semantic = {
        semantic_joint_name(name): index for index, name in enumerate(target.names)
    }
    foot_pairs = tuple(
        (source_by_semantic[semantic], target_by_semantic[semantic])
        for semantic in ("left_foot", "right_foot")
        if semantic in source_by_semantic and semantic in target_by_semantic
    )
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


def retarget_motion(
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
        "mapped_joint_count": len(profile.transfers),
        "target_joint_count": target_skeleton.joint_count,
        "root_translation_scale": profile.root_translation_scale,
        **contact_diagnostics,
    }
