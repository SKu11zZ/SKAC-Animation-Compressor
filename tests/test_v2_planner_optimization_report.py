from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from tools.run_v2_planner_optimization_report import _svg


class V2PlannerOptimizationReportTests(unittest.TestCase):
    def test_visual_is_static_bilingual_and_concrete(self) -> None:
        report = {
            "passed": True,
            "samples": [
                {
                    "sample": "shortest", "frame_count": 200, "speedup": 2.5,
                    "baseline_seconds": 10.0, "optimized_seconds": 4.0,
                },
                {
                    "sample": "median", "frame_count": 500, "speedup": 2.0,
                    "baseline_seconds": 24.0, "optimized_seconds": 12.0,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.svg"
            path.write_text(_svg(report), encoding="utf-8")
            ElementTree.parse(path)
            text = path.read_text(encoding="utf-8").casefold()
        self.assertIn("规划器优化", text)
        self.assertIn("2.50×", text)
        self.assertNotIn("<script", text)


if __name__ == "__main__":
    unittest.main()
