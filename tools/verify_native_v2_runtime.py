from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

from skac_codec.bvh import loads_bvh
from skac_codec.format import PREFIX, decode_bytes
from skac_codec.format_v2 import encode_v2_bytes
from tools.verify_native_runtime import TEST_BVH


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the SKAC v2 Python decoder with the native chunk reader."
    )
    parser.add_argument("--probe", type=Path, required=True)
    return parser.parse_args(argv)


def _raw_chunks(encoded: bytes) -> bytes:
    fields = PREFIX.unpack_from(encoded)
    metadata_size = int(fields[4])
    payload_size = int(fields[5])
    metadata_start = PREFIX.size
    payload_start = metadata_start + metadata_size
    metadata = json.loads(
        encoded[metadata_start:payload_start].decode("utf-8", errors="strict")
    )
    payload = encoded[payload_start : payload_start + payload_size]
    result = bytearray()
    for chunk in metadata["chunks"]:
        start = int(chunk["offset"])
        end = start + int(chunk["compressed_bytes"])
        raw = zlib.decompress(payload[start:end])
        if len(raw) != int(chunk["raw_bytes"]):
            raise AssertionError("fixture chunk size differs from its directory")
        result.extend(raw)
    return bytes(result)


def verify(probe: Path) -> dict[str, float | int | bool]:
    source = loads_bvh(TEST_BVH)
    encoded = encode_v2_bytes(
        source, min_segment_frames=2, max_segment_frames=3
    )
    decoded = decode_bytes(encoded)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        container_path = root / "motion.skac"
        raw_path = root / "raw-chunks.bin"
        container_path.write_bytes(encoded)
        raw_path.write_bytes(_raw_chunks(encoded))
        completed = subprocess.run(
            [str(probe.resolve()), str(container_path), str(raw_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    native_rotations = np.empty_like(decoded.local_rotations)
    native_translations = np.empty_like(decoded.local_translations)
    poses = 0
    for line in completed.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "info":
            if (
                int(fields[1]) != decoded.frame_count
                or int(fields[2]) != decoded.skeleton.joint_count
                or abs(float(fields[3]) - decoded.frame_time) > 1e-12
            ):
                raise AssertionError("native v2 clip info differs from Python")
        elif fields[0] == "pose":
            frame = int(fields[1])
            joint = int(fields[2])
            values = np.asarray([float(item) for item in fields[3:]], dtype=np.float64)
            native_rotations[frame, joint] = values[[3, 0, 1, 2]]
            native_translations[frame, joint] = values[4:7]
            poses += 1
    expected_poses = decoded.frame_count * decoded.skeleton.joint_count
    if poses != expected_poses:
        raise AssertionError("native v2 probe did not return every pose")
    rotation_difference = float(
        np.max(np.abs(native_rotations - decoded.local_rotations))
    )
    translation_difference = float(
        np.max(np.abs(native_translations - decoded.local_translations))
    )
    if rotation_difference > 1e-6 or translation_difference > 1e-5:
        raise AssertionError("native v2 transforms differ from Python")
    return {
        "passed": True,
        "format_major": struct.unpack_from("<H", encoded, 8)[0],
        "frame_count": decoded.frame_count,
        "joint_count": decoded.skeleton.joint_count,
        "pose_count": expected_poses,
        "rotation_component_difference_max": rotation_difference,
        "translation_component_difference_max": translation_difference,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(verify(args.probe), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
