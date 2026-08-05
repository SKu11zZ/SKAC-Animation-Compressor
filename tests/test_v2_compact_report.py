from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from tests.test_adaptive import adaptive_clip
from tools.report_v2_compact import (
    build_compact_report,
    write_compact_report_json,
    write_compact_report_svg,
)


ROOT = Path(__file__).resolve().parents[1]


class V2CompactReportTests(unittest.TestCase):
    def test_report_records_concrete_sizes_and_quality(self) -> None:
        report = build_compact_report(adaptive_clip())
        containers = {item["label"]: item for item in report["containers"]}
        self.assertLess(
            containers["SKAC v2.1"]["file_bytes"],
            containers["SKAC v2.0"]["file_bytes"],
        )
        self.assertLessEqual(
            report["reconstruction"]["rotation_error_degrees_max"], 0.0625
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_compact_report_json(root / "report.json", report)
            write_compact_report_svg(root / "report.svg", report)
            ElementTree.parse(root / "report.svg")
            svg = (root / "report.svg").read_text(encoding="utf-8")
        self.assertIn("具体文件大小", svg)
        self.assertNotIn("<script", svg.casefold())

    def test_checked_in_report_is_reproducible(self) -> None:
        report = build_compact_report(adaptive_clip())
        stored = json.loads(
            (ROOT / "reports/skac_v2_compact_optimization.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored, report)
        with tempfile.TemporaryDirectory() as temporary:
            rendered = Path(temporary) / "report.svg"
            write_compact_report_svg(rendered, report)
            self.assertEqual(
                rendered.read_text(encoding="utf-8"),
                (ROOT / "reports/skac_v2_compact_optimization.svg").read_text(
                    encoding="utf-8"
                ),
            )


if __name__ == "__main__":
    unittest.main()
