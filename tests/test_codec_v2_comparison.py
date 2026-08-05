from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from skac_codec.format import CodecSettings
from tests.test_codec_cli import SINGLE_JOINT_BVH
from tools.run_codec_v2_comparison import run_comparison, write_report, write_visual


ROOT = Path(__file__).resolve().parents[1]


class CodecV2ComparisonTests(unittest.TestCase):
    def test_paired_runner_writes_path_free_json_and_static_visual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for character in ("One", "Two"):
                character_root = root / character
                character_root.mkdir()
                (character_root / "Move.bvh").write_text(
                    SINGLE_JOINT_BVH, encoding="utf-8"
                )
            report = run_comparison(
                root,
                characters=("One", "Two"),
                animations_per_character=1,
                selected_animations=("Move",),
                decode_iterations=1,
                settings=CodecSettings.preset("high"),
                min_segment_frames=2,
                max_segment_frames=3,
            )
            json_path = root / "comparison.json"
            svg_path = root / "comparison.svg"
            write_report(json_path, report)
            write_visual(svg_path, report)
            restored_text = json_path.read_text(encoding="utf-8")
            restored = json.loads(restored_text)
            svg = svg_path.read_text(encoding="utf-8")
            ElementTree.parse(svg_path)
        self.assertTrue(restored["passed"])
        self.assertEqual(restored["sampling"]["sample_count"], 2)
        self.assertIn("SKAC v1", restored["overall"])
        self.assertIn("SKAC v2.1", restored["overall"])
        self.assertNotIn(str(root), restored_text)
        self.assertIn("同一组", svg)
        self.assertNotIn("<script", svg.casefold())

    def test_checked_public_report_keeps_failures_and_distribution_visible(self) -> None:
        report = json.loads(
            (ROOT / "reports/codec_v1_v2_1_8x20_public.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["sampling"]["sample_count"], 160)
        self.assertEqual(report["comparison"]["clips_smaller_with_v2_1"], 128)
        self.assertEqual(report["comparison"]["clips_larger_with_v2_1"], 32)
        self.assertEqual(report["overall"]["SKAC v1"]["encoded_bytes"], 5_742_173)
        self.assertEqual(report["overall"]["SKAC v2.1"]["encoded_bytes"], 5_256_422)
        failed = [item["id"] for item in report["checks"] if not item["passed"]]
        self.assertEqual(failed, ["skac_v1_global_position_error"])
        ElementTree.parse(ROOT / "reports/codec_v1_v2_1_8x20_public.svg")


if __name__ == "__main__":
    unittest.main()
