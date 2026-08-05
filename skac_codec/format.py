from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .math3d import quaternion_angular_error_degrees, quaternion_slerp
from .model import MotionClip, Skeleton


MAGIC = b"SKACANIM"
VERSION_MAJOR = 1
VERSION_MINOR = 0
FLAG_ZLIB = 1
PREFIX = struct.Struct("<8sHHIIQQII")
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_RAW_PAYLOAD_BYTES = 512 * 1024 * 1024
_SMALLEST_THREE_LIMIT = 1.0 / math.sqrt(2.0)


class SkacFormatError(ValueError):
    pass


@dataclass(frozen=True)
class CodecSettings:
    rotation_bits: int = 16
    translation_bits: int = 20
    rotation_error_degrees: float = 0.05
    translation_error_fraction: float = 1e-5
    zlib_level: int = 9
    quality_name: str = "high"

    def __post_init__(self) -> None:
        if not 8 <= self.rotation_bits <= 20:
            raise ValueError("rotation_bits must be between 8 and 20")
        if not 8 <= self.translation_bits <= 24:
            raise ValueError("translation_bits must be between 8 and 24")
        if not np.isfinite(self.rotation_error_degrees) or self.rotation_error_degrees <= 0:
            raise ValueError("rotation_error_degrees must be positive and finite")
        if not np.isfinite(self.translation_error_fraction) or self.translation_error_fraction <= 0:
            raise ValueError("translation_error_fraction must be positive and finite")
        if not 0 <= self.zlib_level <= 9:
            raise ValueError("zlib_level must be between 0 and 9")
        if not self.quality_name:
            raise ValueError("quality_name cannot be empty")

    @classmethod
    def preset(cls, name: str) -> "CodecSettings":
        presets = {
            "low": cls(10, 12, 1.0, 1e-3, 6, "low"),
            "medium": cls(13, 16, 0.25, 1e-4, 9, "medium"),
            "high": cls(16, 20, 0.05, 1e-5, 9, "high"),
        }
        try:
            return presets[name.casefold()]
        except KeyError as error:
            raise ValueError("quality must be low, medium, or high") from error


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.accumulator = 0
        self.count = 0

    def write(self, value: int, bits: int) -> None:
        if value < 0 or value >= (1 << bits):
            raise ValueError("packed integer is outside its bit width")
        self.accumulator |= int(value) << self.count
        self.count += bits
        while self.count >= 8:
            self.data.append(self.accumulator & 0xFF)
            self.accumulator >>= 8
            self.count -= 8

    def finish(self) -> bytes:
        if self.count:
            self.data.append(self.accumulator & 0xFF)
            self.accumulator = 0
            self.count = 0
        return bytes(self.data)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0
        self.accumulator = 0
        self.count = 0

    def read(self, bits: int) -> int:
        while self.count < bits:
            if self.offset >= len(self.data):
                raise SkacFormatError("packed payload ended unexpectedly")
            self.accumulator |= self.data[self.offset] << self.count
            self.offset += 1
            self.count += 8
        value = self.accumulator & ((1 << bits) - 1)
        self.accumulator >>= bits
        self.count -= bits
        return value


class _ByteCursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.data):
            raise SkacFormatError("track payload ended unexpectedly")
        result = self.data[self.offset : self.offset + count]
        self.offset += count
        return result

    def uint32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def varuint(self) -> int:
        value = 0
        shift = 0
        for _ in range(5):
            byte = self.take(1)[0]
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7
        raise SkacFormatError("variable-length integer exceeds 32 bits")

    def require_finished(self) -> None:
        if self.offset != len(self.data):
            raise SkacFormatError("track payload contains trailing bytes")


def _encode_varuint(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError("variable-length integer is outside uint32")
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


def _encode_indices(indices: np.ndarray) -> bytes:
    result = bytearray()
    previous = 0
    for position, index in enumerate(np.asarray(indices, dtype=np.int64)):
        delta = int(index) if position == 0 else int(index) - previous
        if delta < 0:
            raise ValueError("keyframe indices must be sorted")
        result.extend(_encode_varuint(delta))
        previous = int(index)
    return bytes(result)


def _decode_indices(cursor: _ByteCursor, count: int, frame_count: int) -> np.ndarray:
    if count <= 0 or count > frame_count:
        raise SkacFormatError("keyframe count is outside the frame range")
    result = np.empty(count, dtype=np.int64)
    previous = 0
    for position in range(count):
        delta = cursor.varuint()
        index = delta if position == 0 else previous + delta
        if index >= frame_count or (position > 0 and index <= previous):
            raise SkacFormatError("keyframe indices are not strictly increasing")
        result[position] = index
        previous = index
    if result[0] != 0:
        raise SkacFormatError("every track must begin at frame zero")
    return result


def _packed_byte_count(value_count: int, bits_per_value: int) -> int:
    return (value_count * bits_per_value + 7) // 8


def _pack_unsigned(values: Iterable[int], bits: int) -> bytes:
    writer = _BitWriter()
    for value in values:
        writer.write(int(value), bits)
    return writer.finish()


def _unpack_unsigned(data: bytes, count: int, bits: int) -> np.ndarray:
    expected = _packed_byte_count(count, bits)
    if len(data) != expected:
        raise SkacFormatError(f"packed field has {len(data)} bytes, expected {expected}")
    reader = _BitReader(data)
    return np.fromiter((reader.read(bits) for _ in range(count)), dtype=np.uint32, count=count)


def _rotation_joint_indices(skeleton: Skeleton) -> tuple[int, ...]:
    return tuple(
        joint
        for joint, channels in enumerate(skeleton.channels)
        if any(channel.endswith("rotation") for channel in channels)
    )


def _rotation_key_indices(track: np.ndarray, threshold_degrees: float) -> np.ndarray:
    frame_count = track.shape[0]
    if frame_count == 1:
        return np.asarray([0], dtype=np.int64)
    constant_error = quaternion_angular_error_degrees(
        track, np.broadcast_to(track[0], track.shape)
    )
    if float(np.max(constant_error)) <= threshold_degrees:
        return np.asarray([0], dtype=np.int64)

    kept = {0, frame_count - 1}
    pending = [(0, frame_count - 1)]
    while pending:
        start, end = pending.pop()
        if end - start <= 1:
            continue
        frames = np.arange(start + 1, end, dtype=np.int64)
        amount = (frames - start) / (end - start)
        predicted = quaternion_slerp(track[start], track[end], amount)
        errors = quaternion_angular_error_degrees(track[frames], predicted)
        relative = int(np.argmax(errors))
        if float(errors[relative]) > threshold_degrees:
            selected = int(frames[relative])
            kept.add(selected)
            pending.append((start, selected))
            pending.append((selected, end))
    return np.asarray(sorted(kept), dtype=np.int64)


def _scalar_key_indices(track: np.ndarray, threshold: float) -> np.ndarray:
    frame_count = track.shape[0]
    if frame_count == 1 or float(np.max(np.abs(track - track[0]))) <= threshold:
        return np.asarray([0], dtype=np.int64)
    kept = {0, frame_count - 1}
    pending = [(0, frame_count - 1)]
    while pending:
        start, end = pending.pop()
        if end - start <= 1:
            continue
        frames = np.arange(start + 1, end, dtype=np.int64)
        amount = (frames - start) / (end - start)
        predicted = track[start] + amount * (track[end] - track[start])
        errors = np.abs(track[frames] - predicted)
        relative = int(np.argmax(errors))
        if float(errors[relative]) > threshold:
            selected = int(frames[relative])
            kept.add(selected)
            pending.append((start, selected))
            pending.append((selected, end))
    return np.asarray(sorted(kept), dtype=np.int64)


def _interpolate_rotation_track(
    frame_count: int, indices: np.ndarray, values: np.ndarray
) -> np.ndarray:
    if len(indices) == 1:
        return np.broadcast_to(values[0], (frame_count, 4)).copy()
    frames = np.arange(frame_count, dtype=np.int64)
    segments = np.searchsorted(indices, frames, side="right") - 1
    segments = np.clip(segments, 0, len(indices) - 2)
    starts = indices[segments]
    ends = indices[segments + 1]
    amount = (frames - starts) / (ends - starts)
    return quaternion_slerp(values[segments], values[segments + 1], amount)


def _interpolate_scalar_track(
    frame_count: int, indices: np.ndarray, values: np.ndarray
) -> np.ndarray:
    if len(indices) == 1:
        return np.full(frame_count, values[0], dtype=np.float64)
    frames = np.arange(frame_count, dtype=np.int64)
    segments = np.searchsorted(indices, frames, side="right") - 1
    segments = np.clip(segments, 0, len(indices) - 2)
    starts = indices[segments]
    ends = indices[segments + 1]
    amount = (frames - starts) / (ends - starts)
    return values[segments] + amount * (values[segments + 1] - values[segments])


def _encode_rotations(quaternions: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4).copy()
    lengths = np.linalg.norm(flat, axis=1, keepdims=True)
    flat /= lengths
    omitted = np.argmax(np.abs(flat), axis=1)
    signs = flat[np.arange(flat.shape[0]), omitted] < 0.0
    flat[signs] *= -1.0
    maximum = (1 << bits) - 1
    writer = _BitWriter()
    for quaternion, omitted_index in zip(flat, omitted, strict=True):
        writer.write(int(omitted_index), 2)
        for component, value in enumerate(quaternion):
            if component == omitted_index:
                continue
            normalized = (float(value) + _SMALLEST_THREE_LIMIT) / (
                2.0 * _SMALLEST_THREE_LIMIT
            )
            quantized = int(round(np.clip(normalized, 0.0, 1.0) * maximum))
            writer.write(quantized, bits)
    return writer.finish()


def _decode_rotations(data: bytes, count: int, bits: int) -> np.ndarray:
    expected = _packed_byte_count(count, 2 + 3 * bits)
    if len(data) != expected:
        raise SkacFormatError(f"rotation payload has {len(data)} bytes, expected {expected}")
    maximum = (1 << bits) - 1
    reader = _BitReader(data)
    result = np.empty((count, 4), dtype=np.float64)
    for index in range(count):
        omitted = reader.read(2)
        squared = 0.0
        for component in range(4):
            if component == omitted:
                continue
            quantized = reader.read(bits)
            value = (quantized / maximum) * (2.0 * _SMALLEST_THREE_LIMIT) - _SMALLEST_THREE_LIMIT
            result[index, component] = value
            squared += value * value
        result[index, omitted] = math.sqrt(max(0.0, 1.0 - squared))
    lengths = np.linalg.norm(result, axis=1, keepdims=True)
    return result / lengths


def _encode_rotation_tracks(
    clip: MotionClip,
    joints: tuple[int, ...],
    bits: int,
    threshold_degrees: float,
) -> tuple[bytes, list[int]]:
    result = bytearray()
    key_counts: list[int] = []
    for joint in joints:
        track = clip.local_rotations[:, joint, :]
        indices = _rotation_key_indices(track, threshold_degrees)
        values = track[indices]
        packed = _encode_rotations(values, bits)
        result.extend(struct.pack("<I", len(indices)))
        result.extend(_encode_indices(indices))
        result.extend(packed)
        key_counts.append(len(indices))
    return bytes(result), key_counts


def _decode_rotation_tracks(
    data: bytes,
    frame_count: int,
    skeleton: Skeleton,
    joints: tuple[int, ...],
    bits: int,
    expected_key_counts: list[int],
) -> np.ndarray:
    if len(expected_key_counts) != len(joints):
        raise SkacFormatError("rotation key counts do not match the track table")
    cursor = _ByteCursor(data)
    rotations = np.zeros((frame_count, skeleton.joint_count, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    for track_index, joint in enumerate(joints):
        key_count = cursor.uint32()
        if key_count != expected_key_counts[track_index]:
            raise SkacFormatError("rotation key count does not match metadata")
        indices = _decode_indices(cursor, key_count, frame_count)
        packed_size = _packed_byte_count(key_count, 2 + 3 * bits)
        values = _decode_rotations(cursor.take(packed_size), key_count, bits)
        rotations[:, joint, :] = _interpolate_rotation_track(
            frame_count, indices, values
        )
    cursor.require_finished()
    return rotations


def _encode_translation_tracks(
    clip: MotionClip,
    components: tuple[tuple[int, int], ...],
    bits: int,
    error_fraction: float,
) -> tuple[bytes, list[float], list[float], list[int]]:
    result = bytearray()
    minima: list[float] = []
    maxima: list[float] = []
    key_counts: list[int] = []
    maximum_integer = (1 << bits) - 1
    for joint, axis in components:
        track = clip.local_translations[:, joint, axis]
        lower = float(np.min(track))
        upper = float(np.max(track))
        value_range = upper - lower
        threshold = max(value_range * error_fraction, 1e-12)
        indices = _scalar_key_indices(track, threshold)
        values = track[indices]
        if value_range > 0.0:
            quantized = np.rint(
                np.clip((values - lower) / value_range, 0.0, 1.0) * maximum_integer
            ).astype(np.uint32)
        else:
            quantized = np.zeros(len(values), dtype=np.uint32)
        result.extend(struct.pack("<I", len(indices)))
        result.extend(_encode_indices(indices))
        result.extend(_pack_unsigned(quantized, bits))
        minima.append(lower)
        maxima.append(upper)
        key_counts.append(len(indices))
    return bytes(result), minima, maxima, key_counts


def _decode_translation_tracks(
    data: bytes,
    frame_count: int,
    skeleton: Skeleton,
    components: tuple[tuple[int, int], ...],
    bits: int,
    minima: list[float],
    maxima: list[float],
    expected_key_counts: list[int],
) -> np.ndarray:
    translations = np.broadcast_to(
        skeleton.offsets, (frame_count, skeleton.joint_count, 3)
    ).copy()
    if not components:
        if data or minima or maxima or expected_key_counts:
            raise SkacFormatError("translation metadata is inconsistent")
        return translations
    if not (
        len(minima)
        == len(maxima)
        == len(expected_key_counts)
        == len(components)
    ):
        raise SkacFormatError("translation metadata does not match the track table")
    cursor = _ByteCursor(data)
    maximum_integer = (1 << bits) - 1
    for track_index, (joint, axis) in enumerate(components):
        key_count = cursor.uint32()
        if key_count != expected_key_counts[track_index]:
            raise SkacFormatError("translation key count does not match metadata")
        indices = _decode_indices(cursor, key_count, frame_count)
        packed_size = _packed_byte_count(key_count, bits)
        quantized = _unpack_unsigned(cursor.take(packed_size), key_count, bits)
        lower = minima[track_index]
        upper = maxima[track_index]
        values = lower + (quantized.astype(np.float64) / maximum_integer) * (
            upper - lower
        )
        translations[:, joint, axis] = _interpolate_scalar_track(
            frame_count, indices, values
        )
    cursor.require_finished()
    return translations


def _decompress_limited(payload: bytes, expected_size: int) -> bytes:
    if expected_size < 0 or expected_size > MAX_RAW_PAYLOAD_BYTES:
        raise SkacFormatError("declared raw payload size is outside the safety limit")
    decompressor = zlib.decompressobj()
    result = decompressor.decompress(payload, expected_size + 1)
    if (
        len(result) > expected_size
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise SkacFormatError("compressed payload exceeds its declared size")
    if len(result) != expected_size or not decompressor.eof:
        raise SkacFormatError("compressed payload size does not match its declaration")
    return result


def encode_bytes(clip: MotionClip, settings: CodecSettings | None = None) -> bytes:
    settings = settings or CodecSettings()
    rotation_joints = _rotation_joint_indices(clip.skeleton)
    translation_components = clip.skeleton.animated_translation_components
    rotation_bytes, rotation_key_counts = _encode_rotation_tracks(
        clip,
        rotation_joints,
        settings.rotation_bits,
        settings.rotation_error_degrees,
    )
    translation_bytes, minima, maxima, translation_key_counts = (
        _encode_translation_tracks(
            clip,
            translation_components,
            settings.translation_bits,
            settings.translation_error_fraction,
        )
    )
    raw_payload = rotation_bytes + translation_bytes
    payload = zlib.compress(raw_payload, level=settings.zlib_level)

    metadata: dict[str, Any] = {
        "schema": "skac.animation",
        "schema_version": "1.0.0",
        "frame_count": clip.frame_count,
        "frame_time": clip.frame_time,
        "duration_seconds": clip.duration_seconds,
        "skeleton": clip.skeleton.to_dict(),
        "skeleton_sha256": clip.skeleton.signature(),
        "codec": {
            "name": "key_reduced_smallest_three_v1",
            "quality": settings.quality_name,
            "rotation_bits": settings.rotation_bits,
            "translation_bits": settings.translation_bits,
            "rotation_error_degrees": settings.rotation_error_degrees,
            "translation_error_fraction": settings.translation_error_fraction,
            "rotation_joints": list(rotation_joints),
            "rotation_key_counts": rotation_key_counts,
            "translation_components": [list(item) for item in translation_components],
            "translation_key_counts": translation_key_counts,
            "translation_minima": minima,
            "translation_maxima": maxima,
            "rotation_payload_bytes": len(rotation_bytes),
            "translation_payload_bytes": len(translation_bytes),
            "compression": "zlib",
        },
    }
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the format limit")
    prefix = PREFIX.pack(
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        FLAG_ZLIB,
        len(metadata_bytes),
        len(payload),
        len(raw_payload),
        zlib.crc32(metadata_bytes),
        zlib.crc32(payload),
    )
    return prefix + metadata_bytes + payload


def _read_container(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < PREFIX.size:
        raise SkacFormatError("file is shorter than the SKAC prefix")
    (
        magic,
        major,
        minor,
        flags,
        metadata_size,
        payload_size,
        raw_payload_size,
        metadata_crc,
        payload_crc,
    ) = PREFIX.unpack_from(data)
    if magic != MAGIC:
        raise SkacFormatError("file magic is not SKACANIM")
    if major != VERSION_MAJOR:
        raise SkacFormatError(f"unsupported SKAC major version: {major}")
    if minor > VERSION_MINOR:
        raise SkacFormatError(f"unsupported SKAC minor version: {minor}")
    if flags != FLAG_ZLIB:
        raise SkacFormatError(f"unsupported SKAC flags: {flags}")
    if metadata_size > MAX_METADATA_BYTES:
        raise SkacFormatError("metadata exceeds the safety limit")
    expected_file_size = PREFIX.size + metadata_size + payload_size
    if len(data) != expected_file_size:
        raise SkacFormatError(
            f"file has {len(data)} bytes, expected {expected_file_size}"
        )
    metadata_bytes = data[PREFIX.size : PREFIX.size + metadata_size]
    payload = data[PREFIX.size + metadata_size :]
    if zlib.crc32(metadata_bytes) != metadata_crc:
        raise SkacFormatError("metadata CRC does not match")
    if zlib.crc32(payload) != payload_crc:
        raise SkacFormatError("payload CRC does not match")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SkacFormatError("metadata is not valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise SkacFormatError("metadata root must be an object")
    return metadata, _decompress_limited(payload, int(raw_payload_size))


def decode_bytes(data: bytes) -> MotionClip:
    metadata, raw_payload = _read_container(data)
    if metadata.get("schema") != "skac.animation" or metadata.get("schema_version") != "1.0.0":
        raise SkacFormatError("unsupported animation metadata schema")
    try:
        frame_count = int(metadata["frame_count"])
        frame_time = float(metadata["frame_time"])
        skeleton = Skeleton.from_dict(metadata["skeleton"])
        codec = metadata["codec"]
        codec_name = str(codec["name"])
        rotation_bits = int(codec["rotation_bits"])
        translation_bits = int(codec["translation_bits"])
        rotation_joints = tuple(int(item) for item in codec["rotation_joints"])
        rotation_key_counts = [int(item) for item in codec["rotation_key_counts"]]
        translation_components = tuple(
            (int(item[0]), int(item[1])) for item in codec["translation_components"]
        )
        translation_key_counts = [
            int(item) for item in codec["translation_key_counts"]
        ]
        translation_minima = [float(item) for item in codec["translation_minima"]]
        translation_maxima = [float(item) for item in codec["translation_maxima"]]
        rotation_size = int(codec["rotation_payload_bytes"])
        translation_size = int(codec["translation_payload_bytes"])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise SkacFormatError("animation metadata is incomplete or invalid") from error
    if frame_count <= 0 or frame_count > 100_000_000:
        raise SkacFormatError("frame_count is outside the safety limit")
    if codec_name != "key_reduced_smallest_three_v1":
        raise SkacFormatError(f"unsupported codec payload: {codec_name}")
    if skeleton.signature() != metadata.get("skeleton_sha256"):
        raise SkacFormatError("skeleton signature does not match")
    if not 8 <= rotation_bits <= 20 or not 8 <= translation_bits <= 24:
        raise SkacFormatError("codec bit width is outside the supported range")
    if rotation_joints != _rotation_joint_indices(skeleton):
        raise SkacFormatError("rotation joint table does not match the skeleton")
    if translation_components != skeleton.animated_translation_components:
        raise SkacFormatError("translation table does not match the skeleton")
    if rotation_size < 0 or translation_size < 0 or rotation_size + translation_size != len(raw_payload):
        raise SkacFormatError("payload section lengths are inconsistent")

    rotations = _decode_rotation_tracks(
        raw_payload[:rotation_size],
        frame_count,
        skeleton,
        rotation_joints,
        rotation_bits,
        rotation_key_counts,
    )
    translations = _decode_translation_tracks(
        raw_payload[rotation_size:],
        frame_count,
        skeleton,
        translation_components,
        translation_bits,
        translation_minima,
        translation_maxima,
        translation_key_counts,
    )
    return MotionClip(
        skeleton=skeleton,
        local_rotations=rotations,
        local_translations=translations,
        frame_time=frame_time,
    )


def write_skac(path: Path, clip: MotionClip, settings: CodecSettings | None = None) -> None:
    encoded = encode_bytes(clip, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def read_skac(path: Path) -> MotionClip:
    return decode_bytes(path.read_bytes())


def inspect_bytes(data: bytes) -> dict[str, Any]:
    metadata, _ = _read_container(data)
    codec = metadata.get("codec", {})
    return {
        "file_bytes": len(data),
        "format_major": VERSION_MAJOR,
        "format_minor": VERSION_MINOR,
        "frame_count": metadata.get("frame_count"),
        "frame_time": metadata.get("frame_time"),
        "duration_seconds": metadata.get("duration_seconds"),
        "joint_count": len(metadata.get("skeleton", {}).get("names", [])),
        "skeleton_sha256": metadata.get("skeleton_sha256"),
        "codec": {
            "name": codec.get("name"),
            "quality": codec.get("quality"),
            "rotation_bits": codec.get("rotation_bits"),
            "translation_bits": codec.get("translation_bits"),
            "rotation_error_degrees": codec.get("rotation_error_degrees"),
            "translation_error_fraction": codec.get("translation_error_fraction"),
            "rotation_keyframes": sum(codec.get("rotation_key_counts", [])),
            "translation_keyframes": sum(codec.get("translation_key_counts", [])),
            "compression": codec.get("compression"),
            "rotation_payload_bytes": codec.get("rotation_payload_bytes"),
            "translation_payload_bytes": codec.get("translation_payload_bytes"),
        },
    }


def inspect_file(path: Path) -> dict[str, Any]:
    return inspect_bytes(path.read_bytes())
