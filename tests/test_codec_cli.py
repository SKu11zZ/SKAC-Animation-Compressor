from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from skac_codec.bvh import loads_bvh
from skac_codec.cli import main


SINGLE_JOINT_BVH = """HIERARCHY
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
Frames: 3
Frame Time: 0.0333333333
0 0 0 0 0 0
1 0 0 5 0 0
2 0 0 10 0 0
"""


class CodecCliTests(unittest.TestCase):
    def test_encode_inspect_decode_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bvh"
            encoded = root / "motion.skac"
            restored = root / "restored.bvh"
            profile = root / "target.profile.json"
            retargeted = root / "retargeted.bvh"
            source.write_text(SINGLE_JOINT_BVH, encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(["encode", str(source), "-o", str(encoded), "--quality", "high"]),
                    0,
                )
            report = json.loads(output.getvalue())
            self.assertEqual(report["command"], "encode")
            self.assertEqual(report["frame_count"], 3)
            self.assertTrue(encoded.is_file())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["inspect", str(encoded)]), 0)
            self.assertEqual(json.loads(output.getvalue())["format_major"], 1)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["decode", str(encoded), "-o", str(restored)]), 0)
            self.assertEqual(loads_bvh(restored.read_text(encoding="utf-8")).frame_count, 3)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "profile",
                            str(encoded),
                            str(source),
                            "-o",
                            str(profile),
                        ]
                    ),
                    0,
                )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "decode",
                            str(encoded),
                            "--target",
                            str(source),
                            "--profile",
                            str(profile),
                            "-o",
                            str(retargeted),
                        ]
                    ),
                    0,
                )
            self.assertEqual(json.loads(output.getvalue())["command"], "decode-retarget")
            self.assertEqual(loads_bvh(retargeted.read_text(encoding="utf-8")).frame_count, 3)


if __name__ == "__main__":
    unittest.main()
