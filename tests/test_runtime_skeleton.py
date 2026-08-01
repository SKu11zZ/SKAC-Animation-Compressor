from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skac_codec.bvh import loads_bvh
from skac_codec.runtime import runtime_skeleton_bytes, save_runtime_skeleton


BVH = """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  End Site
  {
    OFFSET 0 1 0
  }
}
MOTION
Frames: 1
Frame Time: 0.0333333333
0 0 0 0 0 0
"""


class RuntimeSkeletonTests(unittest.TestCase):
    def test_document_is_deterministic_and_hash_pinned(self) -> None:
        skeleton = loads_bvh(BVH).skeleton
        first = runtime_skeleton_bytes(skeleton)
        second = runtime_skeleton_bytes(skeleton)
        self.assertEqual(first, second)
        value = json.loads(first)
        self.assertEqual(value["schema"], "skac.runtime_skeleton")
        self.assertEqual(value["skeleton_sha256"], skeleton.signature())

    def test_save_uses_public_json_only(self) -> None:
        skeleton = loads_bvh(BVH).skeleton
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.json"
            save_runtime_skeleton(path, skeleton)
            value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["skeleton"]["names"], ["Root"])


if __name__ == "__main__":
    unittest.main()
