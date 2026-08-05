from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .format import SkacFormatError, inspect_bytes


MAGIC = b"SKACPACK"
VERSION_MAJOR = 1
VERSION_MINOR = 0
SCHEMA = "skac.pack"
SCHEMA_VERSION = "1.0.0"
PREFIX = struct.Struct("<8sHHIIQII")
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_ENTRY_COUNT = 1_000_000
_ENTRY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SkacPackError(ValueError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _validate_entry_id(entry_id: str) -> str:
    if not isinstance(entry_id, str) or not _ENTRY_ID.fullmatch(entry_id):
        raise SkacPackError(
            "entry id must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return entry_id


@dataclass(frozen=True)
class SkacPack:
    metadata: dict[str, Any]
    payload: bytes

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.metadata["entries"])

    def entry_bytes(self, entry_id: str) -> bytes:
        _validate_entry_id(entry_id)
        entries = {
            str(item["id"]): str(item["blob_sha256"])
            for item in self.metadata["entries"]
        }
        try:
            digest = entries[entry_id]
        except KeyError as error:
            raise SkacPackError(f"pack does not contain entry: {entry_id}") from error
        blobs = {
            str(item["sha256"]): (int(item["offset"]), int(item["bytes"]))
            for item in self.metadata["blobs"]
        }
        offset, size = blobs[digest]
        return self.payload[offset : offset + size]


def encode_pack(entries: Mapping[str, bytes]) -> bytes:
    if not entries:
        raise SkacPackError("pack requires at least one entry")
    if len(entries) > MAX_ENTRY_COUNT:
        raise SkacPackError("pack entry count exceeds the safety limit")

    normalized: list[tuple[str, bytes, dict[str, Any], str]] = []
    for entry_id, encoded in entries.items():
        name = _validate_entry_id(entry_id)
        if not isinstance(encoded, bytes):
            raise SkacPackError(f"entry {name} must contain immutable bytes")
        try:
            details = inspect_bytes(encoded)
        except SkacFormatError as error:
            raise SkacPackError(f"entry {name} is not a valid .skac animation") from error
        digest = hashlib.sha256(encoded).hexdigest()
        normalized.append((name, encoded, details, digest))
    normalized.sort(key=lambda item: item[0])
    if len({item[0] for item in normalized}) != len(normalized):
        raise SkacPackError("pack entry ids must be unique")

    unique_blobs: dict[str, bytes] = {}
    for _, encoded, _, digest in normalized:
        previous = unique_blobs.setdefault(digest, encoded)
        if previous != encoded:
            raise SkacPackError("SHA-256 collision detected between pack entries")

    payload = bytearray()
    blob_records: list[dict[str, Any]] = []
    for digest in sorted(unique_blobs):
        encoded = unique_blobs[digest]
        offset = len(payload)
        payload.extend(encoded)
        blob_records.append(
            {
                "bytes": len(encoded),
                "offset": offset,
                "sha256": digest,
            }
        )
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise SkacPackError("pack payload exceeds the safety limit")

    entry_records = []
    for name, _, details, digest in normalized:
        entry_records.append(
            {
                "blob_sha256": digest,
                "duration_seconds": details["duration_seconds"],
                "frame_count": details["frame_count"],
                "frame_time": details["frame_time"],
                "id": name,
                "skeleton_sha256": details["skeleton_sha256"],
            }
        )
    metadata = {
        "blob_count": len(blob_records),
        "entries": entry_records,
        "entry_count": len(entry_records),
        "blobs": blob_records,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
    }
    metadata_bytes = _canonical_json(metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise SkacPackError("pack metadata exceeds the safety limit")
    prefix = PREFIX.pack(
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        0,
        len(metadata_bytes),
        len(payload),
        zlib.crc32(metadata_bytes),
        zlib.crc32(payload),
    )
    return prefix + metadata_bytes + bytes(payload)


def decode_pack(data: bytes) -> SkacPack:
    if len(data) < PREFIX.size:
        raise SkacPackError("file is shorter than the SKAC Pack prefix")
    (
        magic,
        major,
        minor,
        flags,
        metadata_size,
        payload_size,
        metadata_crc,
        payload_crc,
    ) = PREFIX.unpack_from(data)
    if magic != MAGIC:
        raise SkacPackError("file magic is not SKACPACK")
    if major != VERSION_MAJOR or minor > VERSION_MINOR:
        raise SkacPackError(f"unsupported SKAC Pack version: {major}.{minor}")
    if flags != 0:
        raise SkacPackError(f"unsupported SKAC Pack flags: {flags}")
    if metadata_size > MAX_METADATA_BYTES or payload_size > MAX_PAYLOAD_BYTES:
        raise SkacPackError("pack declaration exceeds the safety limit")
    expected_size = PREFIX.size + metadata_size + payload_size
    if len(data) != expected_size:
        raise SkacPackError(f"file has {len(data)} bytes, expected {expected_size}")
    metadata_bytes = data[PREFIX.size : PREFIX.size + metadata_size]
    payload = data[PREFIX.size + metadata_size :]
    if zlib.crc32(metadata_bytes) != metadata_crc:
        raise SkacPackError("pack metadata CRC does not match")
    if zlib.crc32(payload) != payload_crc:
        raise SkacPackError("pack payload CRC does not match")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SkacPackError("pack metadata is not valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise SkacPackError("pack metadata root must be an object")
    _validate_metadata(metadata, payload)
    return SkacPack(metadata=metadata, payload=payload)


def _validate_metadata(metadata: dict[str, Any], payload: bytes) -> None:
    if metadata.get("schema") != SCHEMA or metadata.get("schema_version") != SCHEMA_VERSION:
        raise SkacPackError("unsupported SKAC Pack metadata schema")
    entries = metadata.get("entries")
    blobs = metadata.get("blobs")
    if not isinstance(entries, list) or not isinstance(blobs, list):
        raise SkacPackError("pack entry and blob tables must be arrays")
    if not entries or len(entries) > MAX_ENTRY_COUNT:
        raise SkacPackError("pack entry count is outside the safety limit")
    if not blobs:
        raise SkacPackError("pack blob table cannot be empty")
    if (
        type(metadata.get("entry_count")) is not int
        or type(metadata.get("blob_count")) is not int
        or metadata["entry_count"] != len(entries)
        or metadata["blob_count"] != len(blobs)
    ):
        raise SkacPackError("pack table counts do not match metadata")

    blob_map: dict[str, tuple[int, int]] = {}
    blob_details: dict[str, dict[str, Any]] = {}
    cursor = 0
    for item in blobs:
        if not isinstance(item, dict):
            raise SkacPackError("pack blob record must be an object")
        try:
            digest = item["sha256"]
            offset = item["offset"]
            size = item["bytes"]
        except KeyError as error:
            raise SkacPackError("pack blob record is incomplete") from error
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise SkacPackError("pack blob SHA-256 is invalid")
        if type(offset) is not int or type(size) is not int:
            raise SkacPackError("pack blob offset and size must be integers")
        if digest in blob_map or offset != cursor or size <= 0:
            raise SkacPackError("pack blob table is duplicated, sparse, or unordered")
        end = offset + size
        if end > len(payload):
            raise SkacPackError("pack blob extends beyond the payload")
        encoded = payload[offset:end]
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise SkacPackError("pack blob SHA-256 does not match")
        try:
            details = inspect_bytes(encoded)
        except SkacFormatError as error:
            raise SkacPackError("pack blob is not a valid .skac animation") from error
        blob_map[digest] = (offset, size)
        blob_details[digest] = details
        cursor = end
    if cursor != len(payload):
        raise SkacPackError("pack payload contains unindexed bytes")
    if list(blob_map) != sorted(blob_map):
        raise SkacPackError("pack blob table is not in canonical order")

    seen_ids: set[str] = set()
    entry_order: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            raise SkacPackError("pack entry record must be an object")
        try:
            entry_id = _validate_entry_id(item["id"])
            digest = item["blob_sha256"]
            frame_count = item["frame_count"]
            frame_time = item["frame_time"]
            duration = item["duration_seconds"]
            skeleton_sha256 = item["skeleton_sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise SkacPackError("pack entry record is incomplete") from error
        if (
            not isinstance(digest, str)
            or type(frame_count) is not int
            or isinstance(frame_time, bool)
            or not isinstance(frame_time, (int, float))
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not isinstance(skeleton_sha256, str)
        ):
            raise SkacPackError("pack entry record has invalid field types")
        if entry_id in seen_ids or digest not in blob_map:
            raise SkacPackError("pack entry id is duplicated or references a missing blob")
        seen_ids.add(entry_id)
        entry_order.append(entry_id)
        details = blob_details[digest]
        if (
            frame_count != details["frame_count"]
            or frame_time != details["frame_time"]
            or duration != details["duration_seconds"]
            or skeleton_sha256 != details["skeleton_sha256"]
        ):
            raise SkacPackError("pack entry metadata does not match its animation blob")
    if entry_order != sorted(entry_order):
        raise SkacPackError("pack entry table is not in canonical order")


def inspect_pack_bytes(data: bytes) -> dict[str, Any]:
    archive = decode_pack(data)
    entries = archive.metadata["entries"]
    return {
        "file_bytes": len(data),
        "format_major": VERSION_MAJOR,
        "format_minor": VERSION_MINOR,
        "entry_count": len(entries),
        "unique_blob_count": len(archive.metadata["blobs"]),
        "deduplicated_entry_count": len(entries) - len(archive.metadata["blobs"]),
        "payload_bytes": len(archive.payload),
        "entries": entries,
    }


def inspect_pack_file(path: Path) -> dict[str, Any]:
    return inspect_pack_bytes(path.read_bytes())


def read_pack(path: Path) -> SkacPack:
    return decode_pack(path.read_bytes())


def write_pack(path: Path, entries: Mapping[str, bytes]) -> None:
    encoded = encode_pack(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
