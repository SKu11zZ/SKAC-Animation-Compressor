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
            runtime_target = root / "target.runtime.json"
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
                            "runtime-skeleton",
                            str(source),
                            "-o",
                            str(runtime_target),
                        ]
                    ),
                    0,
                )
            self.assertEqual(json.loads(output.getvalue())["joint_count"], 1)
            self.assertTrue(runtime_target.is_file())
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

    def test_quality_gate_cli_routes_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bvh"
            source.write_text(SINGLE_JOINT_BVH, encoding="utf-8")

            same_json = root / "same.json"
            same_svg = root / "same.svg"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "quality-gate-same",
                            str(source),
                            "--output",
                            str(same_json),
                            "--visual",
                            str(same_svg),
                            "--decode-iterations",
                            "1",
                        ]
                    ),
                    0,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["command"], "quality-gate-same")
            self.assertEqual(result["evaluation_case"], "same_character")

            different_json = root / "different.json"
            different_svg = root / "different.svg"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "quality-gate-different",
                            str(source),
                            str(source),
                            "--output",
                            str(different_json),
                            "--visual",
                            str(different_svg),
                            "--minimum-core-coverage",
                            "0",
                            "--decode-iterations",
                            "1",
                            "--pipeline-iterations",
                            "1",
                            "--frame-samples",
                            "10",
                        ]
                    ),
                    0,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["command"], "quality-gate-different")
            self.assertEqual(result["evaluation_case"], "different_character")

    def test_pack_create_inspect_and_extract_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bvh"
            encoded = root / "motion.skac"
            archive = root / "motions.skacpack"
            extracted = root / "restored.skac"
            source.write_text(SINGLE_JOINT_BVH, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["encode", str(source), "-o", str(encoded)]), 0
                )
                self.assertEqual(
                    main(
                        [
                            "pack-create",
                            "--clip",
                            f"idle={encoded}",
                            "--clip",
                            f"idle-copy={encoded}",
                            "-o",
                            str(archive),
                        ]
                    ),
                    0,
                )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["pack-inspect", str(archive)]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["entry_count"], 2)
            self.assertEqual(report["unique_blob_count"], 1)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "pack-extract",
                            str(archive),
                            "idle",
                            "-o",
                            str(extracted),
                        ]
                    ),
                    0,
                )
            self.assertEqual(extracted.read_bytes(), encoded.read_bytes())

    def test_adaptive_plan_writes_json_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bvh"
            report = root / "adaptive.json"
            visual = root / "adaptive.svg"
            source.write_text(SINGLE_JOINT_BVH, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "adaptive-plan",
                            str(source),
                            "-o",
                            str(report),
                            "--visual",
                            str(visual),
                        ]
                    ),
                    0,
                )
            result = json.loads(output.getvalue())
            self.assertTrue(result["passed"])
            self.assertTrue(report.is_file())
            self.assertIn("感知规划", visual.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
