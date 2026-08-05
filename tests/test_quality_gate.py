from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from skac_codec.format import CodecSettings
from skac_codec.quality import (
    run_quality_gate,
    write_quality_report_json,
    write_quality_report_svg,
)
from tests.test_retarget import source_clip, target_skeleton


class QualityGateTests(unittest.TestCase):
    def test_gate_writes_json_and_self_contained_svg(self) -> None:
        report = run_quality_gate(
            source_clip(),
            target_skeleton(1.2, True),
            settings=CodecSettings.preset("high"),
            decode_iterations=2,
            pipeline_iterations=2,
            frame_samples=30,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["evaluation_case"], "different_character")
        self.assertEqual(report["profile"]["schema_version"], "2.0.0")
        self.assertGreater(report["performance"]["retarget_fps_at_p95"], 60.0)
        self.assertFalse(report["scope"]["retarget_ground_truth_available"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "gate.json"
            svg_path = root / "gate.svg"
            write_quality_report_json(json_path, report)
            write_quality_report_svg(svg_path, report)
            restored = json.loads(json_path.read_text(encoding="utf-8"))
            svg = svg_path.read_text(encoding="utf-8")
            ElementTree.parse(svg_path)
        self.assertTrue(restored["passed"])
        self.assertIn("SKAC v1 - Different character - compiled Profile playback Gate", svg)
        self.assertNotIn("<script", svg.casefold())

    def test_same_character_gate_bypasses_profile_runtime(self) -> None:
        source = source_clip()
        report = run_quality_gate(
            source,
            source.skeleton,
            settings=CodecSettings.preset("high"),
            decode_iterations=2,
            evaluation_case="same_character",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["evaluation_case"], "same_character")
        self.assertIsNone(report["target"])
        self.assertIsNone(report["profile"])
        self.assertIsNone(report["runtime_equivalence"])
        self.assertIsNone(report["runtime_diagnostics"])
        self.assertNotIn("retarget_frame_ms_p95", report["performance"])
        check_ids = {item["id"] for item in report["checks"]}
        self.assertNotIn("profile_core_coverage", check_ids)
        self.assertNotIn("pipeline_realtime_factor", check_ids)

    def test_v2_gate_uses_the_chunked_codec_and_labels_the_visual(self) -> None:
        source = source_clip()
        report = run_quality_gate(
            source,
            source.skeleton,
            settings=CodecSettings.preset("high"),
            decode_iterations=1,
            evaluation_case="same_character",
            format_version=2,
            min_segment_frames=2,
            max_segment_frames=3,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["codec"]["format_version"], 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2.svg"
            write_quality_report_svg(path, report)
            svg = path.read_text(encoding="utf-8")
        self.assertIn("SKAC v2 - Same character - direct Codec decode Gate", svg)

    def test_same_character_gate_rejects_a_different_skeleton(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical source and target skeletons"):
            run_quality_gate(
                source_clip(),
                target_skeleton(1.2, True),
                evaluation_case="same_character",
            )


if __name__ == "__main__":
    unittest.main()
