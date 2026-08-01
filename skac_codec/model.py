from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

_CHANNELS = {
    "Xposition", "Yposition", "Zposition",
    "Xrotation", "Yrotation", "Zrotation",
}


def _finite_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class Skeleton:
    names: tuple[str, ...]
    parents: IntArray
    offsets: FloatArray
    channels: tuple[tuple[str, ...], ...]
    end_site_offsets: FloatArray
    has_end_sites: BoolArray

    def __post_init__(self) -> None:
        names = tuple(str(name).strip() for name in self.names)
        if not names or any(not name for name in names):
            raise ValueError("skeleton requires non-empty joint names")
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")

        count = len(names)
        parents = np.asarray(self.parents, dtype=np.int32)
        offsets = _finite_array(self.offsets, (count, 3), "offsets")
        end_offsets = _finite_array(
            self.end_site_offsets, (count, 3), "end_site_offsets"
        )
        has_end_sites = np.asarray(self.has_end_sites, dtype=np.bool_)
        channels = tuple(tuple(str(channel) for channel in item) for item in self.channels)

        if parents.shape != (count,) or has_end_sites.shape != (count,):
            raise ValueError("skeleton arrays must match the joint count")
        if len(channels) != count:
            raise ValueError("channels must match the joint count")
        if parents[0] != -1:
            raise ValueError("root parent must be -1")
        for index, parent in enumerate(parents[1:], start=1):
            if parent < 0 or parent >= index:
                raise ValueError("parents must precede their children")

        for joint_channels in channels:
            if len(set(joint_channels)) != len(joint_channels):
                raise ValueError("a joint cannot repeat a BVH channel")
            if any(channel not in _CHANNELS for channel in joint_channels):
                raise ValueError("unsupported BVH channel")
            rotation_axes = [channel[0] for channel in joint_channels if channel.endswith("rotation")]
            if rotation_axes and sorted(rotation_axes) != ["X", "Y", "Z"]:
                raise ValueError("rotated joints require exactly three unique rotation axes")

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "parents", parents.copy())
        object.__setattr__(self, "offsets", offsets.copy())
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "end_site_offsets", end_offsets.copy())
        object.__setattr__(self, "has_end_sites", has_end_sites.copy())
        self.parents.setflags(write=False)
        self.offsets.setflags(write=False)
        self.end_site_offsets.setflags(write=False)
        self.has_end_sites.setflags(write=False)

    @property
    def joint_count(self) -> int:
        return len(self.names)

    @property
    def animated_translation_components(self) -> tuple[tuple[int, int], ...]:
        axes = {"X": 0, "Y": 1, "Z": 2}
        return tuple(
            (joint, axes[channel[0]])
            for joint, joint_channels in enumerate(self.channels)
            for channel in joint_channels
            if channel.endswith("position")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "parents": self.parents.tolist(),
            "offsets": self.offsets.tolist(),
            "channels": [list(item) for item in self.channels],
            "end_site_offsets": self.end_site_offsets.tolist(),
            "has_end_sites": self.has_end_sites.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Skeleton":
        return cls(
            names=tuple(str(item) for item in value["names"]),
            parents=np.asarray(value["parents"], dtype=np.int32),
            offsets=np.asarray(value["offsets"], dtype=np.float64),
            channels=tuple(tuple(str(channel) for channel in item) for item in value["channels"]),
            end_site_offsets=np.asarray(value["end_site_offsets"], dtype=np.float64),
            has_end_sites=np.asarray(value["has_end_sites"], dtype=np.bool_),
        )

    def signature(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MotionClip:
    skeleton: Skeleton
    local_rotations: FloatArray
    local_translations: FloatArray
    frame_time: float

    def __post_init__(self) -> None:
        rotations = np.asarray(self.local_rotations, dtype=np.float64)
        translations = np.asarray(self.local_translations, dtype=np.float64)
        expected_rotations = (rotations.shape[0], self.skeleton.joint_count, 4)
        expected_translations = (rotations.shape[0], self.skeleton.joint_count, 3)
        if rotations.ndim != 3 or rotations.shape != expected_rotations:
            raise ValueError("local_rotations must have shape (frames, joints, 4)")
        if translations.shape != expected_translations:
            raise ValueError("local_translations must have shape (frames, joints, 3)")
        if rotations.shape[0] == 0:
            raise ValueError("motion clip requires at least one frame")
        if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
            raise ValueError("motion arrays must contain finite values")
        lengths = np.linalg.norm(rotations, axis=-1, keepdims=True)
        if np.any(lengths <= 1e-12):
            raise ValueError("local_rotations contains a zero quaternion")
        if not np.isfinite(self.frame_time) or self.frame_time <= 0:
            raise ValueError("frame_time must be positive and finite")

        rotations = rotations / lengths
        object.__setattr__(self, "local_rotations", rotations.copy())
        object.__setattr__(self, "local_translations", translations.copy())
        object.__setattr__(self, "frame_time", float(self.frame_time))
        self.local_rotations.setflags(write=False)
        self.local_translations.setflags(write=False)

    @property
    def frame_count(self) -> int:
        return int(self.local_rotations.shape[0])

    @property
    def duration_seconds(self) -> float:
        return max(0, self.frame_count - 1) * self.frame_time

    @property
    def raw_channel_bytes_float32(self) -> int:
        channel_count = sum(len(item) for item in self.skeleton.channels)
        return self.frame_count * channel_count * 4
