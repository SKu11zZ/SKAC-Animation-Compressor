from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def canonical_joint_name(name: str) -> str:
    """Return a namespace-independent public semantic joint name."""

    value = name.rsplit(":", 1)[-1].strip()
    if not value:
        raise ValueError("joint name cannot be empty")
    return value


def _float_array(value: ArrayLike, shape_suffix: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < len(shape_suffix) or array.shape[-len(shape_suffix) :] != shape_suffix:
        raise ValueError(f"{name} has an invalid shape")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _normalize_quaternions(value: ArrayLike, name: str) -> FloatArray:
    quaternions = _float_array(value, (4,), name)
    lengths = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError(f"{name} contains a zero quaternion")
    return quaternions / lengths


@dataclass(frozen=True)
class Skeleton:
    names: tuple[str, ...]
    parents: IntArray
    offsets: FloatArray
    height: float

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        parents = np.asarray(self.parents, dtype=np.int64)
        offsets = _float_array(self.offsets, (3,), "offsets")
        if not names:
            raise ValueError("skeleton cannot be empty")
        if parents.shape != (len(names),) or offsets.shape != (len(names), 3):
            raise ValueError("skeleton arrays must match the joint count")
        if parents[0] != -1:
            raise ValueError("root parent must be -1")
        for index, parent in enumerate(parents[1:], start=1):
            if parent < 0 or parent >= index:
                raise ValueError("parents must precede their children")
        if not np.isfinite(self.height) or self.height <= 0:
            raise ValueError("skeleton height must be positive and finite")
        canonical = [canonical_joint_name(name) for name in names]
        if len(set(canonical)) != len(canonical):
            raise ValueError("canonical joint names must be unique")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "parents", parents.copy())
        object.__setattr__(self, "offsets", offsets.copy())
        self.parents.setflags(write=False)
        self.offsets.setflags(write=False)


@dataclass(frozen=True)
class Motion:
    local_rotations: FloatArray
    root_positions: FloatArray

    def __post_init__(self) -> None:
        rotations = _normalize_quaternions(self.local_rotations, "local_rotations")
        positions = _float_array(self.root_positions, (3,), "root_positions")
        if rotations.ndim != 3 or positions.ndim != 2:
            raise ValueError("motion arrays must have shapes (frames, joints, 4) and (frames, 3)")
        if rotations.shape[0] == 0 or rotations.shape[0] != positions.shape[0]:
            raise ValueError("motion arrays must have the same non-zero frame count")
        object.__setattr__(self, "local_rotations", rotations.copy())
        object.__setattr__(self, "root_positions", positions.copy())
        self.local_rotations.setflags(write=False)
        self.root_positions.setflags(write=False)


@dataclass(frozen=True)
class RetargetProfile:
    source_joint_count: int
    target_joint_count: int
    target_to_source: tuple[tuple[int, int], ...]
    root_translation_scale: float
    configuration_scope: str = "source_target_skeleton_pair"
    per_animation_adjustment: bool = False

    def __post_init__(self) -> None:
        if self.source_joint_count <= 0 or self.target_joint_count <= 0:
            raise ValueError("joint counts must be positive")
        if not np.isfinite(self.root_translation_scale) or self.root_translation_scale <= 0:
            raise ValueError("root translation scale must be positive and finite")
        seen_targets: set[int] = set()
        for target, source in self.target_to_source:
            if not 0 <= source < self.source_joint_count:
                raise ValueError("source mapping index is outside the skeleton")
            if not 0 <= target < self.target_joint_count:
                raise ValueError("target mapping index is outside the skeleton")
            if target in seen_targets:
                raise ValueError("each target joint may be mapped only once")
            seen_targets.add(target)
        if self.per_animation_adjustment:
            raise ValueError("public automatic profiles cannot be animation-specific")


def build_profile(source: Skeleton, target: Skeleton) -> RetargetProfile:
    """Freeze an exact semantic-name mapping for a source-target skeleton pair."""

    source_indices = {
        canonical_joint_name(name): index for index, name in enumerate(source.names)
    }
    mapping = tuple(
        (target_index, source_indices[canonical_joint_name(target_name)])
        for target_index, target_name in enumerate(target.names)
        if canonical_joint_name(target_name) in source_indices
    )
    if not mapping or mapping[0] != (0, 0):
        raise ValueError("source and target roots must share the same semantic name")
    return RetargetProfile(
        source_joint_count=len(source.names),
        target_joint_count=len(target.names),
        target_to_source=mapping,
        root_translation_scale=target.height / source.height,
    )


def retarget_motion(profile: RetargetProfile, source: Motion) -> Motion:
    """Retarget by frozen semantic rotation transfer and height-scaled root motion."""

    if source.local_rotations.shape[1] != profile.source_joint_count:
        raise ValueError("source motion joint count does not match the frozen profile")
    frames = source.local_rotations.shape[0]
    target_rotations = np.zeros((frames, profile.target_joint_count, 4), dtype=np.float64)
    target_rotations[..., 0] = 1.0
    for target_index, source_index in profile.target_to_source:
        target_rotations[:, target_index, :] = source.local_rotations[:, source_index, :]
    return Motion(
        local_rotations=target_rotations,
        root_positions=source.root_positions * profile.root_translation_scale,
    )
