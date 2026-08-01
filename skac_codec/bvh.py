from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .math3d import euler_order_to_quaternion, quaternion_to_euler_order
from .model import MotionClip, Skeleton


_TOKEN_PATTERN = re.compile(r"[{}]|[^\s{}]+")
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


class BvhFormatError(ValueError):
    pass


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _TOKEN_PATTERN.findall(text)
        self.position = 0
        self.names: list[str] = []
        self.parents: list[int] = []
        self.offsets: list[list[float]] = []
        self.channels: list[tuple[str, ...]] = []
        self.end_site_offsets: list[list[float]] = []
        self.has_end_sites: list[bool] = []

    def pop(self) -> str:
        if self.position >= len(self.tokens):
            raise BvhFormatError("unexpected end of BVH file")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def expect(self, expected: str) -> None:
        actual = self.pop()
        if actual != expected:
            raise BvhFormatError(f"expected {expected!r}, got {actual!r}")

    def floats(self, count: int) -> list[float]:
        try:
            values = [float(self.pop()) for _ in range(count)]
        except ValueError as error:
            raise BvhFormatError("expected a numeric BVH value") from error
        if not np.isfinite(values).all():
            raise BvhFormatError("BVH contains a non-finite value")
        return values

    def joint(self, kind: str, parent: int) -> int:
        if kind not in {"ROOT", "JOINT"}:
            raise BvhFormatError(f"unsupported hierarchy node: {kind}")
        name = self.pop()
        index = len(self.names)
        self.names.append(name)
        self.parents.append(parent)
        self.offsets.append([0.0, 0.0, 0.0])
        self.channels.append(())
        self.end_site_offsets.append([0.0, 0.0, 0.0])
        self.has_end_sites.append(False)

        self.expect("{")
        saw_offset = False
        saw_channels = False
        while True:
            token = self.pop()
            if token == "}":
                break
            if token == "OFFSET":
                if saw_offset:
                    raise BvhFormatError(f"joint {name!r} repeats OFFSET")
                self.offsets[index] = self.floats(3)
                saw_offset = True
            elif token == "CHANNELS":
                if saw_channels:
                    raise BvhFormatError(f"joint {name!r} repeats CHANNELS")
                try:
                    count = int(self.pop())
                except ValueError as error:
                    raise BvhFormatError("CHANNELS requires an integer count") from error
                if count < 0:
                    raise BvhFormatError("CHANNELS count cannot be negative")
                self.channels[index] = tuple(self.pop() for _ in range(count))
                saw_channels = True
            elif token == "JOINT":
                self.joint(token, index)
            elif token == "End":
                self.expect("Site")
                if self.has_end_sites[index]:
                    raise BvhFormatError(f"joint {name!r} repeats End Site")
                self.expect("{")
                self.expect("OFFSET")
                self.end_site_offsets[index] = self.floats(3)
                self.expect("}")
                self.has_end_sites[index] = True
            else:
                raise BvhFormatError(f"unexpected token in joint {name!r}: {token!r}")
        if not saw_offset or not saw_channels:
            raise BvhFormatError(f"joint {name!r} requires OFFSET and CHANNELS")
        return index

    def parse(self) -> MotionClip:
        self.expect("HIERARCHY")
        self.expect("ROOT")
        self.joint("ROOT", -1)
        self.expect("MOTION")

        frames_token = self.pop()
        if frames_token not in {"Frames:", "Frames"}:
            raise BvhFormatError("expected Frames header")
        if frames_token == "Frames":
            self.expect(":")
        try:
            frame_count = int(self.pop())
        except ValueError as error:
            raise BvhFormatError("Frames requires an integer") from error
        if frame_count <= 0:
            raise BvhFormatError("BVH requires at least one frame")

        self.expect("Frame")
        time_token = self.pop()
        if time_token not in {"Time:", "Time"}:
            raise BvhFormatError("expected Frame Time header")
        if time_token == "Time":
            self.expect(":")
        try:
            frame_time = float(self.pop())
        except ValueError as error:
            raise BvhFormatError("Frame Time requires a number") from error
        if not np.isfinite(frame_time) or frame_time <= 0:
            raise BvhFormatError("Frame Time must be positive and finite")

        channel_count = sum(len(item) for item in self.channels)
        expected_values = frame_count * channel_count
        remaining = self.tokens[self.position :]
        if len(remaining) != expected_values:
            raise BvhFormatError(
                f"motion payload has {len(remaining)} values, expected {expected_values}"
            )
        try:
            values = np.asarray(remaining, dtype=np.float64).reshape(frame_count, channel_count)
        except ValueError as error:
            raise BvhFormatError("motion payload contains a non-numeric value") from error
        if not np.isfinite(values).all():
            raise BvhFormatError("motion payload contains a non-finite value")

        skeleton = Skeleton(
            names=tuple(self.names),
            parents=np.asarray(self.parents, dtype=np.int32),
            offsets=np.asarray(self.offsets, dtype=np.float64),
            channels=tuple(self.channels),
            end_site_offsets=np.asarray(self.end_site_offsets, dtype=np.float64),
            has_end_sites=np.asarray(self.has_end_sites, dtype=np.bool_),
        )
        translations = np.broadcast_to(
            skeleton.offsets, (frame_count, skeleton.joint_count, 3)
        ).copy()
        rotations = np.zeros((frame_count, skeleton.joint_count, 4), dtype=np.float64)
        rotations[..., 0] = 1.0

        column = 0
        for joint, joint_channels in enumerate(skeleton.channels):
            rotation_columns: list[int] = []
            rotation_order = ""
            for channel in joint_channels:
                current = values[:, column]
                if channel.endswith("position"):
                    translations[:, joint, _AXIS_INDEX[channel[0]]] += current
                else:
                    rotation_order += channel[0]
                    rotation_columns.append(column)
                column += 1
            if rotation_columns:
                angles = np.radians(values[:, rotation_columns])
                rotations[:, joint, :] = euler_order_to_quaternion(rotation_order, angles)

        return MotionClip(
            skeleton=skeleton,
            local_rotations=rotations,
            local_translations=translations,
            frame_time=frame_time,
        )


def loads_bvh(text: str) -> MotionClip:
    return _Parser(text).parse()


def read_bvh(path: Path) -> MotionClip:
    return loads_bvh(path.read_text(encoding="utf-8-sig", errors="strict"))


def _format_number(value: float) -> str:
    if abs(value) < 5e-12:
        value = 0.0
    return f"{value:.9g}"


def dumps_bvh(clip: MotionClip) -> str:
    skeleton = clip.skeleton
    children: list[list[int]] = [[] for _ in range(skeleton.joint_count)]
    for child, parent in enumerate(skeleton.parents):
        if parent >= 0:
            children[int(parent)].append(child)

    lines = ["HIERARCHY"]

    def emit_joint(joint: int, depth: int) -> None:
        indent = "\t" * depth
        kind = "ROOT" if joint == 0 else "JOINT"
        lines.append(f"{indent}{kind} {skeleton.names[joint]}")
        lines.append(f"{indent}{{")
        offset = " ".join(_format_number(item) for item in skeleton.offsets[joint])
        lines.append(f"{indent}\tOFFSET {offset}")
        channels = " ".join(skeleton.channels[joint])
        suffix = f" {channels}" if channels else ""
        lines.append(f"{indent}\tCHANNELS {len(skeleton.channels[joint])}{suffix}")
        for child in children[joint]:
            emit_joint(child, depth + 1)
        if skeleton.has_end_sites[joint]:
            end_offset = " ".join(
                _format_number(item) for item in skeleton.end_site_offsets[joint]
            )
            lines.extend(
                (
                    f"{indent}\tEnd Site",
                    f"{indent}\t{{",
                    f"{indent}\t\tOFFSET {end_offset}",
                    f"{indent}\t}}",
                )
            )
        lines.append(f"{indent}}}")

    emit_joint(0, 0)
    lines.extend(
        (
            "MOTION",
            f"Frames: {clip.frame_count}",
            f"Frame Time: {_format_number(clip.frame_time)}",
        )
    )

    euler_degrees: dict[int, np.ndarray] = {}
    for joint, joint_channels in enumerate(skeleton.channels):
        order = "".join(
            channel[0] for channel in joint_channels if channel.endswith("rotation")
        )
        if order:
            radians = quaternion_to_euler_order(clip.local_rotations[:, joint, :], order)
            euler_degrees[joint] = np.degrees(np.unwrap(radians, axis=0))

    for frame in range(clip.frame_count):
        row: list[str] = []
        for joint, joint_channels in enumerate(skeleton.channels):
            rotation_component = 0
            for channel in joint_channels:
                if channel.endswith("position"):
                    axis = _AXIS_INDEX[channel[0]]
                    value = clip.local_translations[frame, joint, axis] - skeleton.offsets[joint, axis]
                else:
                    value = euler_degrees[joint][frame, rotation_component]
                    rotation_component += 1
                row.append(_format_number(float(value)))
        lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


def write_bvh(path: Path, clip: MotionClip) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_bvh(clip), encoding="utf-8", newline="\n")
