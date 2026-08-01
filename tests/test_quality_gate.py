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
        self.assertIn("SKAC Codec Runtime Quality Gate", svg)
        self.assertNotIn("<script", svg.casefold())


if __name__ == "__main__":
    unittest.main()
