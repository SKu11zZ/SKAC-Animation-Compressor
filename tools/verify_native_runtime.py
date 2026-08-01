from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from skac_codec.bvh import loads_bvh
from skac_codec.format import CodecSettings, _read_container, decode_bytes, encode_bytes


TEST_BVH = """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Child
  {
    OFFSET 0 1 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 1 0
    }
  }
}
MOTION
Frames: 4
Frame Time: 0.0333333333
0 0 0 0 0 0 0 0 0
1 0 0 10 0 0 0 5 0
2 1 0 20 5 0 0 10 0
3 1 1 30 10 5 5 15 0
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the public Python decoder with a compiled native probe."
    )
    parser.add_argument("--probe", type=Path, required=True)
    return parser.parse_args(argv)


def verify(probe: Path) -> dict[str, float | int | bool]:
    source = loads_bvh(TEST_BVH)
    encoded = encode_bytes(source, CodecSettings.preset("high"))
    decoded = decode_bytes(encoded)
    metadata, raw_payload = _read_container(encoded)
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata_path = root / "metadata.json"
        payload_path = root / "payload.bin"
        metadata_path.write_bytes(metadata_bytes)
        payload_path.write_bytes(raw_payload)
        completed = subprocess.run(
            [str(probe.resolve()), str(metadata_path), str(payload_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    native_rotations = np.empty_like(decoded.local_rotations)
    native_translations = np.empty_like(decoded.local_translations)
    seen_poses = 0
    for line in completed.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "info":
            if int(fields[1]) != decoded.frame_count:
                raise AssertionError("native frame count differs from Python")
            if int(fields[2]) != decoded.skeleton.joint_count:
                raise AssertionError("native joint count differs from Python")
            if abs(float(fields[3]) - decoded.frame_time) > 1e-12:
                raise AssertionError("native frame time differs from Python")
        elif fields[0] == "joint":
            joint = int(fields[1])
            if int(fields[2]) != int(decoded.skeleton.parents[joint]):
                raise AssertionError("native joint parent differs from Python")
            if " ".join(fields[3:]) != decoded.skeleton.names[joint]:
                raise AssertionError("native joint name differs from Python")
        elif fields[0] == "pose":
            frame = int(fields[1])
            joint = int(fields[2])
            values = np.asarray([float(item) for item in fields[3:]], dtype=np.float64)
            native_rotations[frame, joint] = values[[3, 0, 1, 2]]
            native_translations[frame, joint] = values[4:7]
            seen_poses += 1

    expected_poses = decoded.frame_count * decoded.skeleton.joint_count
    if seen_poses != expected_poses:
        raise AssertionError("native probe did not return every pose")
    rotation_difference = float(
        np.max(np.abs(native_rotations - decoded.local_rotations))
    )
    translation_difference = float(
        np.max(np.abs(native_translations - decoded.local_translations))
    )
    if rotation_difference > 1e-6 or translation_difference > 1e-5:
        raise AssertionError("native decoded transforms differ from Python")
    return {
        "passed": True,
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
