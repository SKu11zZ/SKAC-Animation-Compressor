from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive import build_adaptive_plan
from .format import (
    MAGIC,
    MAX_METADATA_BYTES,
    MAX_RAW_PAYLOAD_BYTES,
    PREFIX,
    CodecSettings,
    SkacFormatError,
    _ByteCursor,
    _decode_indices,
    _decode_rotations,
    _decompress_limited,
    _encode_indices,
    _encode_rotations,
    _encode_varuint,
    _interpolate_rotation_track,
    _interpolate_scalar_track,
    _pack_unsigned,
    _packed_byte_count,
    _rotation_joint_indices,
    _unpack_unsigned,
)
from .model import MotionClip, Skeleton


VERSION_MAJOR = 2
VERSION_MINOR = 1
FLAG_CHUNKED_ZLIB = 2
LEGACY_SCHEMA_VERSION = "2.0.0"
SCHEMA_VERSION = "2.1.0"
SEGMENT_INDEX_SCHEMA = "skac.segment_index"
SEGMENT_INDEX_VERSION = "1.0.0"
CODEC_NAME = "skac_v2_adaptive_segmented"
_KNOWN_REQUIRED_ROLES = {
    "base.skeleton",
    "segment.index",
    "base.segment",
    "base.rotation",
    "base.translation",
}


@dataclass(frozen=True)
class _Chunk:
    identifier: str
    role: str
    required: bool
    offset: int
    compressed_bytes: int
    raw_bytes: int
    crc32: int
    raw_crc32: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _rotation_segment_bytes(
    clip: MotionClip, start: int, end: int, tracks: list[dict[str, Any]]
) -> bytes:
    result = bytearray(b"RSEG")
    result.extend(struct.pack("<HI", 1, len(tracks)))
    for track in tracks:
        joint = int(track["joint"])
        bits = int(track["bits"])
        indices = np.asarray(
            track["retained_frames_relative_to_segment"], dtype=np.int64
        )
        if not 8 <= bits <= 20:
            raise ValueError("planned rotation bit width is outside the v2 range")
        if len(indices) != int(track["key_count"]):
            raise ValueError("planned rotation key count is inconsistent")
        values = clip.local_rotations[start:end, joint][indices]
        result.extend(struct.pack("<IBI", joint, bits, len(indices)))
        result.extend(_encode_indices(indices))
        result.extend(_encode_rotations(values, bits))
    return bytes(result)


def _translation_segment_bytes(
    clip: MotionClip, start: int, end: int, tracks: list[dict[str, Any]]
) -> bytes:
    result = bytearray(b"TSEG")
    result.extend(struct.pack("<HI", 1, len(tracks)))
    for track in tracks:
        joint = int(track["joint"])
        axis = int(track["axis"])
        bits = int(track["bits"])
        indices = np.asarray(
            track["retained_frames_relative_to_segment"], dtype=np.int64
        )
        lower = float(track["minimum"])
        upper = float(track["maximum"])
        if not 0 <= axis <= 2 or not 8 <= bits <= 24:
            raise ValueError("planned translation track is outside the v2 range")
        if len(indices) != int(track["key_count"]):
            raise ValueError("planned translation key count is inconsistent")
        values = clip.local_translations[start:end, joint, axis][indices]
        maximum_integer = (1 << bits) - 1
        if upper > lower:
            quantized = np.rint(
                np.clip((values - lower) / (upper - lower), 0.0, 1.0)
                * maximum_integer
            ).astype(np.uint32)
        else:
            quantized = np.zeros(len(indices), dtype=np.uint32)
        result.extend(struct.pack("<IBBIdd", joint, axis, bits, len(indices), lower, upper))
        result.extend(_encode_indices(indices))
        result.extend(_pack_unsigned(quantized, bits))
    return bytes(result)


def _compress_chunk(
    identifier: str,
    role: str,
    raw: bytes,
    offset: int,
    zlib_level: int,
) -> tuple[dict[str, Any], bytes]:
    compressed = zlib.compress(raw, level=zlib_level)
    record = {
        "id": identifier,
        "role": role,
        "required": True,
        "offset": offset,
        "compressed_bytes": len(compressed),
        "raw_bytes": len(raw),
        "crc32": zlib.crc32(compressed),
        "raw_crc32": zlib.crc32(raw),
        "compression": "zlib",
    }
    return record, compressed


def _encode_v2_0_bytes(
    clip: MotionClip,
    settings: CodecSettings | None = None,
    *,
    min_segment_frames: int = 8,
    max_segment_frames: int = 32,
) -> bytes:
    settings = settings or CodecSettings.preset("high")
    plan, _ = build_adaptive_plan(
        clip,
        settings=settings,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
    )
    if not plan["passed"]:
        failed = ", ".join(
            item["id"] for item in plan["checks"] if not item["passed"]
        )
        raise ValueError(f"SKAC v2 quality gate failed: {failed}")

    segments = [
        {
            "index": int(item["index"]),
            "start_frame": int(item["start_frame"]),
            "end_frame_exclusive": int(item["end_frame_exclusive"]),
            "rotation_chunk": f'base.rotation.{int(item["index"]):06d}',
            "translation_chunk": f'base.translation.{int(item["index"]):06d}',
        }
        for item in plan["segmentation"]["segments"]
    ]
    segment_index = {
        "schema": SEGMENT_INDEX_SCHEMA,
        "schema_version": SEGMENT_INDEX_VERSION,
        "segments": segments,
    }
    raw_chunks: list[tuple[str, str, bytes]] = [
        ("segment.index", "segment.index", _canonical_json(segment_index))
    ]
    for segment in segments:
        index = int(segment["index"])
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        rotation_tracks = [
            item for item in plan["rotation_tracks"] if int(item["segment"]) == index
        ]
        translation_tracks = [
            item
            for item in plan["translation_tracks"]
            if int(item["segment"]) == index
        ]
        raw_chunks.extend(
            [
                (
                    str(segment["rotation_chunk"]),
                    "base.rotation",
                    _rotation_segment_bytes(clip, start, end, rotation_tracks),
                ),
                (
                    str(segment["translation_chunk"]),
                    "base.translation",
                    _translation_segment_bytes(clip, start, end, translation_tracks),
                ),
            ]
        )

    directory: list[dict[str, Any]] = []
    compressed_chunks: list[bytes] = []
    offset = 0
    for identifier, role, raw in raw_chunks:
        record, compressed = _compress_chunk(
            identifier, role, raw, offset, settings.zlib_level
        )
        directory.append(record)
        compressed_chunks.append(compressed)
        offset += len(compressed)
    payload = b"".join(compressed_chunks)
    raw_payload_bytes = sum(len(raw) for _, _, raw in raw_chunks)

    metadata: dict[str, Any] = {
        "schema": "skac.animation",
        "schema_version": LEGACY_SCHEMA_VERSION,
        "frame_count": clip.frame_count,
        "frame_time": clip.frame_time,
        "duration_seconds": clip.duration_seconds,
        "skeleton": clip.skeleton.to_dict(),
        "skeleton_sha256": clip.skeleton.signature(),
        "codec": {
            "name": CODEC_NAME,
            "quality": settings.quality_name,
            "plan_sha256": plan["plan_sha256"],
            "min_segment_frames": min_segment_frames,
            "max_segment_frames": max_segment_frames,
            "rotation_joints": list(_rotation_joint_indices(clip.skeleton)),
            "translation_components": [
                list(item) for item in clip.skeleton.animated_translation_components
            ],
            "rotation_keyframes": plan["summary"]["planned_rotation_key_count"],
            "translation_keyframes": plan["summary"]["planned_translation_key_count"],
            "rotation_bits_weighted_mean": plan["summary"][
                "planned_rotation_bits_weighted_mean"
            ],
            "translation_bits_weighted_mean": plan["summary"][
                "planned_translation_bits_weighted_mean"
            ],
            "compression": "chunked-zlib",
        },
        "segments": segments,
        "chunks": directory,
    }
    metadata_bytes = _canonical_json(metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the format limit")
    if raw_payload_bytes > MAX_RAW_PAYLOAD_BYTES:
        raise ValueError("raw payload exceeds the format limit")
    prefix = PREFIX.pack(
        MAGIC,
        VERSION_MAJOR,
        0,
        FLAG_CHUNKED_ZLIB,
        len(metadata_bytes),
        len(payload),
        raw_payload_bytes,
        zlib.crc32(metadata_bytes),
        zlib.crc32(payload),
    )
    return prefix + metadata_bytes + payload


def _compact_rotation_payload(
    clip: MotionClip, start: int, end: int, tracks: list[dict[str, Any]]
) -> bytes:
    result = bytearray()
    for track in tracks:
        joint = int(track["joint"])
        bits = int(track["bits"])
        indices = np.asarray(
            track["retained_frames_relative_to_segment"], dtype=np.int64
        )
        if not 8 <= bits <= 20 or len(indices) != int(track["key_count"]):
            raise ValueError("planned compact rotation track is inconsistent")
        result.append(bits)
        result.extend(_encode_varuint(len(indices)))
        result.extend(_encode_indices(indices))
        result.extend(_encode_rotations(clip.local_rotations[start:end, joint][indices], bits))
    return bytes(result)


def _compact_translation_payload(
    clip: MotionClip, start: int, end: int, tracks: list[dict[str, Any]]
) -> bytes:
    result = bytearray()
    for track in tracks:
        joint = int(track["joint"])
        axis = int(track["axis"])
        bits = int(track["bits"])
        indices = np.asarray(
            track["retained_frames_relative_to_segment"], dtype=np.int64
        )
        lower = float(track["minimum"])
        upper = float(track["maximum"])
        if (
            not 0 <= axis <= 2
            or not 8 <= bits <= 24
            or len(indices) != int(track["key_count"])
        ):
            raise ValueError("planned compact translation track is inconsistent")
        values = clip.local_translations[start:end, joint, axis][indices]
        maximum_integer = (1 << bits) - 1
        if upper > lower:
            quantized = np.rint(
                np.clip((values - lower) / (upper - lower), 0.0, 1.0)
                * maximum_integer
            ).astype(np.uint32)
        else:
            quantized = np.zeros(len(indices), dtype=np.uint32)
        result.append(bits)
        result.extend(_encode_varuint(len(indices)))
        result.extend(struct.pack("<dd", lower, upper))
        result.extend(_encode_indices(indices))
        result.extend(_pack_unsigned(quantized, bits))
    return bytes(result)


def _compact_segment_bytes(
    clip: MotionClip,
    start: int,
    end: int,
    rotation_tracks: list[dict[str, Any]],
    translation_tracks: list[dict[str, Any]],
) -> bytes:
    rotation = _compact_rotation_payload(clip, start, end, rotation_tracks)
    translation = _compact_translation_payload(clip, start, end, translation_tracks)
    return (
        b"BSEG"
        + struct.pack("<HII", 1, len(rotation), len(translation))
        + rotation
        + translation
    )


def _compact_segment_index(segment_lengths: list[int]) -> bytes:
    result = bytearray(b"SIDX")
    result.extend(struct.pack("<H", 1))
    result.extend(_encode_varuint(len(segment_lengths)))
    for frame_count in segment_lengths:
        result.extend(_encode_varuint(frame_count))
    return bytes(result)


def encode_v2_bytes(
    clip: MotionClip,
    settings: CodecSettings | None = None,
    *,
    min_segment_frames: int = 8,
    max_segment_frames: int = 32,
) -> bytes:
    settings = settings or CodecSettings.preset("high")
    plan, _ = build_adaptive_plan(
        clip,
        settings=settings,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
    )
    if not plan["passed"]:
        failed = ", ".join(
            item["id"] for item in plan["checks"] if not item["passed"]
        )
        raise ValueError(f"SKAC v2 quality gate failed: {failed}")

    segment_records = plan["segmentation"]["segments"]
    segment_lengths = [int(item["frame_count"]) for item in segment_records]
    raw_chunks: list[tuple[int, bytes]] = [
        (0, _canonical_json(clip.skeleton.to_dict())),
        (1, _compact_segment_index(segment_lengths)),
    ]
    for segment in segment_records:
        index = int(segment["index"])
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        raw_chunks.append(
            (
                2,
                _compact_segment_bytes(
                    clip,
                    start,
                    end,
                    [
                        item
                        for item in plan["rotation_tracks"]
                        if int(item["segment"]) == index
                    ],
                    [
                        item
                        for item in plan["translation_tracks"]
                        if int(item["segment"]) == index
                    ],
                ),
            )
        )

    directory: list[list[int]] = []
    compressed_chunks: list[bytes] = []
    for role, raw in raw_chunks:
        compressed = zlib.compress(raw, level=settings.zlib_level)
        directory.append(
            [
                role,
                1,
                len(compressed),
                len(raw),
                zlib.crc32(compressed),
                zlib.crc32(raw),
            ]
        )
        compressed_chunks.append(compressed)
    payload = b"".join(compressed_chunks)
    raw_payload_bytes = sum(len(raw) for _, raw in raw_chunks)
    metadata: dict[str, Any] = {
        "schema": "skac.animation",
        "schema_version": SCHEMA_VERSION,
        "frame_count": clip.frame_count,
        "frame_time": clip.frame_time,
        "skeleton_sha256": clip.skeleton.signature(),
        "codec": {
            "name": CODEC_NAME,
            "quality": settings.quality_name,
            "plan_sha256": plan["plan_sha256"],
            "rotation_keyframes": plan["summary"]["planned_rotation_key_count"],
            "translation_keyframes": plan["summary"]["planned_translation_key_count"],
            "rotation_bits_weighted_mean": plan["summary"][
                "planned_rotation_bits_weighted_mean"
            ],
            "translation_bits_weighted_mean": plan["summary"][
                "planned_translation_bits_weighted_mean"
            ],
            "compression": "chunked-zlib",
        },
        "chunk_roles": ["base.skeleton", "segment.index", "base.segment"],
        "chunks": directory,
    }
    metadata_bytes = _canonical_json(metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the format limit")
    if raw_payload_bytes > MAX_RAW_PAYLOAD_BYTES:
        raise ValueError("raw payload exceeds the format limit")
    prefix = PREFIX.pack(
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        FLAG_CHUNKED_ZLIB,
        len(metadata_bytes),
        len(payload),
        raw_payload_bytes,
        zlib.crc32(metadata_bytes),
        zlib.crc32(payload),
    )
    return prefix + metadata_bytes + payload


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkacFormatError(f"{label} must be an integer")
    return value


def _parse_chunks(
    value: Any, payload_size: int, role_names: Any = None
) -> tuple[_Chunk, ...]:
    if not isinstance(value, list) or not value:
        raise SkacFormatError("v2 chunk directory must be a non-empty array")
    chunks: list[_Chunk] = []
    identifiers: set[str] = set()
    expected_offset = 0
    compact_roles: list[str] | None = None
    if role_names is not None:
        if (
            not isinstance(role_names, list)
            or not role_names
            or not all(isinstance(item, str) and item for item in role_names)
            or len(set(role_names)) != len(role_names)
        ):
            raise SkacFormatError("v2 compact role table is invalid")
        compact_roles = role_names
    for index, item in enumerate(value):
        if compact_roles is not None:
            if not isinstance(item, list) or len(item) != 6:
                raise SkacFormatError("v2 compact chunk entry must contain six integers")
            values = [
                _integer(value, f"chunks[{index}]") for value in item
            ]
            role_index, required_integer, compressed_bytes, raw_bytes, crc, raw_crc = values
            if not 0 <= role_index < len(compact_roles) or required_integer not in (0, 1):
                raise SkacFormatError("v2 compact chunk role or flags are invalid")
            identifier = f"chunk.{index:06d}"
            role = compact_roles[role_index]
            required = bool(required_integer)
            compression = "zlib"
            chunk = _Chunk(
                identifier=identifier,
                role=role,
                required=required,
                offset=expected_offset,
                compressed_bytes=compressed_bytes,
                raw_bytes=raw_bytes,
                crc32=crc,
                raw_crc32=raw_crc,
            )
        else:
            if not isinstance(item, dict):
                raise SkacFormatError("v2 chunk directory entry must be an object")
            try:
                identifier = item["id"]
                role = item["role"]
                required = item["required"]
                compression = item["compression"]
                chunk = _Chunk(
                    identifier=str(identifier),
                    role=str(role),
                    required=bool(required),
                    offset=_integer(item["offset"], f"chunks[{index}].offset"),
                    compressed_bytes=_integer(
                        item["compressed_bytes"], f"chunks[{index}].compressed_bytes"
                    ),
                    raw_bytes=_integer(item["raw_bytes"], f"chunks[{index}].raw_bytes"),
                    crc32=_integer(item["crc32"], f"chunks[{index}].crc32"),
                    raw_crc32=_integer(
                        item["raw_crc32"], f"chunks[{index}].raw_crc32"
                    ),
                )
            except KeyError as error:
                raise SkacFormatError("v2 chunk directory entry is incomplete") from error
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise SkacFormatError("v2 chunk ids must be unique non-empty strings")
        if not isinstance(role, str) or not role:
            raise SkacFormatError("v2 chunk role must be a non-empty string")
        if not isinstance(required, bool) or compression != "zlib":
            raise SkacFormatError("v2 chunk flags or compression are invalid")
        if chunk.required and chunk.role not in _KNOWN_REQUIRED_ROLES:
            raise SkacFormatError(f"unknown required v2 chunk role: {chunk.role}")
        if (
            chunk.offset != expected_offset
            or chunk.compressed_bytes <= 0
            or chunk.raw_bytes < 0
            or chunk.raw_bytes > MAX_RAW_PAYLOAD_BYTES
            or not 0 <= chunk.crc32 <= 0xFFFFFFFF
            or not 0 <= chunk.raw_crc32 <= 0xFFFFFFFF
        ):
            raise SkacFormatError("v2 chunk directory range is invalid")
        expected_offset += chunk.compressed_bytes
        identifiers.add(identifier)
        chunks.append(chunk)
    if expected_offset != payload_size:
        raise SkacFormatError("v2 chunk directory does not cover the payload")
    return tuple(chunks)


def _parse_container(data: bytes) -> tuple[dict[str, Any], bytes, tuple[_Chunk, ...]]:
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
    if magic != MAGIC or major != VERSION_MAJOR or minor > VERSION_MINOR:
        raise SkacFormatError("container is not a supported SKAC v2 file")
    if flags != FLAG_CHUNKED_ZLIB:
        raise SkacFormatError("unsupported SKAC v2 flags")
    if metadata_size > MAX_METADATA_BYTES or raw_payload_size > MAX_RAW_PAYLOAD_BYTES:
        raise SkacFormatError("v2 container declarations exceed safety limits")
    expected_size = PREFIX.size + metadata_size + payload_size
    if len(data) != expected_size:
        raise SkacFormatError(f"file has {len(data)} bytes, expected {expected_size}")
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
    if metadata.get("schema") != "skac.animation" or metadata.get(
        "schema_version"
    ) not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise SkacFormatError("unsupported SKAC v2 metadata schema")
    chunks = _parse_chunks(
        metadata.get("chunks"), len(payload), metadata.get("chunk_roles")
    )
    if sum(chunk.raw_bytes for chunk in chunks) != raw_payload_size:
        raise SkacFormatError("v2 raw chunk sizes do not match the container")
    return metadata, payload, chunks


def _inflate_chunk(payload: bytes, chunk: _Chunk) -> bytes:
    compressed = payload[chunk.offset : chunk.offset + chunk.compressed_bytes]
    if zlib.crc32(compressed) != chunk.crc32:
        raise SkacFormatError(f"v2 chunk CRC does not match: {chunk.identifier}")
    raw = _decompress_limited(compressed, chunk.raw_bytes)
    if zlib.crc32(raw) != chunk.raw_crc32:
        raise SkacFormatError(f"v2 raw chunk CRC does not match: {chunk.identifier}")
    return raw


def _segments(value: Any, frame_count: int, identifiers: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SkacFormatError("v2 segments must be a non-empty array")
    result: list[dict[str, Any]] = []
    expected_start = 0
    seen_chunks: set[str] = set()
    for expected_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SkacFormatError("v2 segment entry must be an object")
        try:
            index = _integer(item["index"], "segment.index")
            start = _integer(item["start_frame"], "segment.start_frame")
            end = _integer(item["end_frame_exclusive"], "segment.end_frame_exclusive")
            rotation_chunk = item["rotation_chunk"]
            translation_chunk = item["translation_chunk"]
        except KeyError as error:
            raise SkacFormatError("v2 segment entry is incomplete") from error
        if (
            index != expected_index
            or start != expected_start
            or end <= start
            or end > frame_count
            or not isinstance(rotation_chunk, str)
            or not isinstance(translation_chunk, str)
            or rotation_chunk not in identifiers
            or translation_chunk not in identifiers
            or rotation_chunk in seen_chunks
            or translation_chunk in seen_chunks
        ):
            raise SkacFormatError("v2 segment directory is inconsistent")
        result.append(
            {
                "index": index,
                "start_frame": start,
                "end_frame_exclusive": end,
                "rotation_chunk": rotation_chunk,
                "translation_chunk": translation_chunk,
            }
        )
        expected_start = end
        seen_chunks.update((rotation_chunk, translation_chunk))
    if expected_start != frame_count:
        raise SkacFormatError("v2 segments do not cover every frame")
    return result


def _decode_rotation_segment(
    raw: bytes,
    frame_count: int,
    expected_joints: tuple[int, ...],
) -> dict[int, np.ndarray]:
    cursor = _ByteCursor(raw)
    if cursor.take(4) != b"RSEG":
        raise SkacFormatError("v2 rotation chunk magic is invalid")
    version, track_count = struct.unpack("<HI", cursor.take(6))
    if version != 1 or track_count != len(expected_joints):
        raise SkacFormatError("v2 rotation chunk header is inconsistent")
    result: dict[int, np.ndarray] = {}
    for expected_joint in expected_joints:
        joint, bits, key_count = struct.unpack("<IBI", cursor.take(9))
        if joint != expected_joint or not 8 <= bits <= 20:
            raise SkacFormatError("v2 rotation track table is inconsistent")
        indices = _decode_indices(cursor, key_count, frame_count)
        packed_size = _packed_byte_count(key_count, 2 + 3 * bits)
        values = _decode_rotations(cursor.take(packed_size), key_count, bits)
        result[joint] = _interpolate_rotation_track(frame_count, indices, values)
    cursor.require_finished()
    return result


def _decode_translation_segment(
    raw: bytes,
    frame_count: int,
    expected_components: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], np.ndarray]:
    cursor = _ByteCursor(raw)
    if cursor.take(4) != b"TSEG":
        raise SkacFormatError("v2 translation chunk magic is invalid")
    version, track_count = struct.unpack("<HI", cursor.take(6))
    if version != 1 or track_count != len(expected_components):
        raise SkacFormatError("v2 translation chunk header is inconsistent")
    result: dict[tuple[int, int], np.ndarray] = {}
    for expected_joint, expected_axis in expected_components:
        joint, axis, bits, key_count, lower, upper = struct.unpack(
            "<IBBIdd", cursor.take(26)
        )
        if (
            (joint, axis) != (expected_joint, expected_axis)
            or not 8 <= bits <= 24
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or upper < lower
        ):
            raise SkacFormatError("v2 translation track table is inconsistent")
        indices = _decode_indices(cursor, key_count, frame_count)
        packed_size = _packed_byte_count(key_count, bits)
        quantized = _unpack_unsigned(cursor.take(packed_size), key_count, bits)
        maximum = (1 << bits) - 1
        values = lower + quantized.astype(np.float64) * ((upper - lower) / maximum)
        result[(joint, axis)] = _interpolate_scalar_track(
            frame_count, indices, values
        )
    cursor.require_finished()
    return result


def _decode_v2_0_bytes(data: bytes) -> MotionClip:
    metadata, payload, chunks = _parse_container(data)
    try:
        frame_count = _integer(metadata["frame_count"], "frame_count")
        frame_time = float(metadata["frame_time"])
        skeleton = Skeleton.from_dict(metadata["skeleton"])
        codec = metadata["codec"]
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise SkacFormatError("SKAC v2 metadata is incomplete or invalid") from error
    if frame_count <= 0 or frame_count > 100_000_000 or not np.isfinite(frame_time) or frame_time <= 0:
        raise SkacFormatError("SKAC v2 timing is outside the safety range")
    if not isinstance(codec, dict) or codec.get("name") != CODEC_NAME:
        raise SkacFormatError("unsupported SKAC v2 codec payload")
    if skeleton.signature() != metadata.get("skeleton_sha256"):
        raise SkacFormatError("skeleton signature does not match")

    by_id = {chunk.identifier: chunk for chunk in chunks}
    identifiers = set(by_id)
    if "segment.index" not in by_id or by_id["segment.index"].role != "segment.index":
        raise SkacFormatError("SKAC v2 segment index chunk is missing")
    segments = _segments(metadata.get("segments"), frame_count, identifiers)
    expected_index = {
        "schema": SEGMENT_INDEX_SCHEMA,
        "schema_version": SEGMENT_INDEX_VERSION,
        "segments": segments,
    }
    try:
        decoded_index = json.loads(
            _inflate_chunk(payload, by_id["segment.index"]).decode(
                "utf-8", errors="strict"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SkacFormatError("SKAC v2 segment index is not valid JSON") from error
    if decoded_index != expected_index:
        raise SkacFormatError("SKAC v2 segment index differs from metadata")

    rotation_joints = _rotation_joint_indices(skeleton)
    translation_components = skeleton.animated_translation_components
    rotations = np.zeros((frame_count, skeleton.joint_count, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    translations = np.broadcast_to(
        skeleton.offsets, (frame_count, skeleton.joint_count, 3)
    ).copy()
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        count = end - start
        rotation_chunk = by_id[str(segment["rotation_chunk"])]
        translation_chunk = by_id[str(segment["translation_chunk"])]
        if rotation_chunk.role != "base.rotation" or translation_chunk.role != "base.translation":
            raise SkacFormatError("SKAC v2 segment points to the wrong chunk role")
        for joint, track in _decode_rotation_segment(
            _inflate_chunk(payload, rotation_chunk), count, rotation_joints
        ).items():
            rotations[start:end, joint] = track
        for (joint, axis), track in _decode_translation_segment(
            _inflate_chunk(payload, translation_chunk), count, translation_components
        ).items():
            translations[start:end, joint, axis] = track
    return MotionClip(skeleton, rotations, translations, frame_time)


def _compact_role_chunk(chunks: tuple[_Chunk, ...], role: str) -> _Chunk:
    matches = [item for item in chunks if item.role == role]
    if len(matches) != 1:
        raise SkacFormatError(f"SKAC v2 requires exactly one {role} chunk")
    return matches[0]


def _compact_segment_lengths(raw: bytes, frame_count: int) -> list[int]:
    cursor = _ByteCursor(raw)
    if cursor.take(4) != b"SIDX" or struct.unpack("<H", cursor.take(2))[0] != 1:
        raise SkacFormatError("SKAC v2 compact segment index is invalid")
    count = cursor.varuint()
    if count <= 0 or count > frame_count:
        raise SkacFormatError("SKAC v2 compact segment count is invalid")
    lengths = [cursor.varuint() for _ in range(count)]
    cursor.require_finished()
    if any(item <= 0 for item in lengths) or sum(lengths) != frame_count:
        raise SkacFormatError("SKAC v2 compact segments do not cover every frame")
    return lengths


def _decode_compact_rotation_payload(
    raw: bytes,
    frame_count: int,
    expected_joints: tuple[int, ...],
) -> dict[int, np.ndarray]:
    cursor = _ByteCursor(raw)
    result: dict[int, np.ndarray] = {}
    for joint in expected_joints:
        bits = cursor.take(1)[0]
        key_count = cursor.varuint()
        if not 8 <= bits <= 20:
            raise SkacFormatError("SKAC v2 compact rotation bits are invalid")
        indices = _decode_indices(cursor, key_count, frame_count)
        packed_size = _packed_byte_count(key_count, 2 + 3 * bits)
        values = _decode_rotations(cursor.take(packed_size), key_count, bits)
        result[joint] = _interpolate_rotation_track(frame_count, indices, values)
    cursor.require_finished()
    return result


def _decode_compact_translation_payload(
    raw: bytes,
    frame_count: int,
    expected_components: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], np.ndarray]:
    cursor = _ByteCursor(raw)
    result: dict[tuple[int, int], np.ndarray] = {}
    for component in expected_components:
        bits = cursor.take(1)[0]
        key_count = cursor.varuint()
        lower, upper = struct.unpack("<dd", cursor.take(16))
        if (
            not 8 <= bits <= 24
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or upper < lower
        ):
            raise SkacFormatError("SKAC v2 compact translation track is invalid")
        indices = _decode_indices(cursor, key_count, frame_count)
        packed_size = _packed_byte_count(key_count, bits)
        quantized = _unpack_unsigned(cursor.take(packed_size), key_count, bits)
        maximum = (1 << bits) - 1
        values = lower + quantized.astype(np.float64) * ((upper - lower) / maximum)
        result[component] = _interpolate_scalar_track(frame_count, indices, values)
    cursor.require_finished()
    return result


def _decode_v2_1_parsed(
    metadata: dict[str, Any], payload: bytes, chunks: tuple[_Chunk, ...]
) -> MotionClip:
    try:
        frame_count = _integer(metadata["frame_count"], "frame_count")
        frame_time = float(metadata["frame_time"])
        codec = metadata["codec"]
    except (KeyError, TypeError, ValueError) as error:
        raise SkacFormatError("SKAC v2 compact metadata is incomplete") from error
    if (
        frame_count <= 0
        or frame_count > 100_000_000
        or not np.isfinite(frame_time)
        or frame_time <= 0
        or not isinstance(codec, dict)
        or codec.get("name") != CODEC_NAME
    ):
        raise SkacFormatError("SKAC v2 compact metadata is invalid")

    skeleton_chunk = _compact_role_chunk(chunks, "base.skeleton")
    index_chunk = _compact_role_chunk(chunks, "segment.index")
    try:
        skeleton_value = json.loads(
            _inflate_chunk(payload, skeleton_chunk).decode("utf-8", errors="strict")
        )
        skeleton = Skeleton.from_dict(skeleton_value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        raise SkacFormatError("SKAC v2 compact skeleton is invalid") from error
    if skeleton.signature() != metadata.get("skeleton_sha256"):
        raise SkacFormatError("skeleton signature does not match")
    lengths = _compact_segment_lengths(
        _inflate_chunk(payload, index_chunk), frame_count
    )
    base_chunks = [item for item in chunks if item.role == "base.segment"]
    if len(base_chunks) != len(lengths):
        raise SkacFormatError("SKAC v2 compact base chunks do not match the segment index")

    rotation_joints = _rotation_joint_indices(skeleton)
    translation_components = skeleton.animated_translation_components
    rotations = np.zeros((frame_count, skeleton.joint_count, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    translations = np.broadcast_to(
        skeleton.offsets, (frame_count, skeleton.joint_count, 3)
    ).copy()
    start = 0
    for count, chunk in zip(lengths, base_chunks, strict=True):
        raw = _inflate_chunk(payload, chunk)
        cursor = _ByteCursor(raw)
        if cursor.take(4) != b"BSEG":
            raise SkacFormatError("SKAC v2 compact base chunk magic is invalid")
        version, rotation_size, translation_size = struct.unpack(
            "<HII", cursor.take(10)
        )
        if version != 1 or rotation_size + translation_size != len(raw) - 14:
            raise SkacFormatError("SKAC v2 compact base chunk sizes are inconsistent")
        rotation_raw = cursor.take(rotation_size)
        translation_raw = cursor.take(translation_size)
        cursor.require_finished()
        end = start + count
        for joint, track in _decode_compact_rotation_payload(
            rotation_raw, count, rotation_joints
        ).items():
            rotations[start:end, joint] = track
        for (joint, axis), track in _decode_compact_translation_payload(
            translation_raw, count, translation_components
        ).items():
            translations[start:end, joint, axis] = track
        start = end
    return MotionClip(skeleton, rotations, translations, frame_time)


def decode_v2_bytes(data: bytes) -> MotionClip:
    metadata, payload, chunks = _parse_container(data)
    if metadata["schema_version"] == LEGACY_SCHEMA_VERSION:
        return _decode_v2_0_bytes(data)
    return _decode_v2_1_parsed(metadata, payload, chunks)


def inspect_v2_bytes(data: bytes) -> dict[str, Any]:
    metadata, _, chunks = _parse_container(data)
    decoded = decode_v2_bytes(data)
    codec = metadata["codec"]
    return {
        "file_bytes": len(data),
        "format_major": VERSION_MAJOR,
        "format_minor": struct.unpack_from("<H", data, 10)[0],
        "frame_count": decoded.frame_count,
        "frame_time": decoded.frame_time,
        "duration_seconds": decoded.duration_seconds,
        "joint_count": decoded.skeleton.joint_count,
        "skeleton_sha256": decoded.skeleton.signature(),
        "codec": {
            "name": codec.get("name"),
            "quality": codec.get("quality"),
            "segment_count": (
                len(metadata["segments"])
                if metadata["schema_version"] == LEGACY_SCHEMA_VERSION
                else sum(item.role == "base.segment" for item in chunks)
            ),
            "chunk_count": len(chunks),
            "rotation_keyframes": codec.get("rotation_keyframes"),
            "translation_keyframes": codec.get("translation_keyframes"),
            "rotation_bits_weighted_mean": codec.get("rotation_bits_weighted_mean"),
            "translation_bits_weighted_mean": codec.get(
                "translation_bits_weighted_mean"
            ),
            "compression": codec.get("compression"),
            "compressed_chunk_bytes": sum(item.compressed_bytes for item in chunks),
            "raw_chunk_bytes": sum(item.raw_bytes for item in chunks),
        },
        "chunks": [
            {
                "id": item.identifier,
                "role": item.role,
                "required": item.required,
                "compressed_bytes": item.compressed_bytes,
                "raw_bytes": item.raw_bytes,
            }
            for item in chunks
        ],
    }


def write_skac_v2(
    path: Path,
    clip: MotionClip,
    settings: CodecSettings | None = None,
    *,
    min_segment_frames: int = 8,
    max_segment_frames: int = 32,
) -> None:
    encoded = encode_v2_bytes(
        clip,
        settings,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
