from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.run_native_runtime_benchmark import generated_fixture, write_visual


class NativeRuntimeBenchmarkTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_cross_skeleton(self) -> None:
        first_source, first_target = generated_fixture()
        second_source, second_target = generated_fixture()
        self.assertEqual(first_source.skeleton.signature(), second_source.skeleton.signature())
        self.assertEqual(first_target.signature(), second_target.signature())
        self.assertEqual(first_source.skeleton.joint_count, 65)
        self.assertEqual(first_target.joint_count, 67)
        self.assertNotEqual(first_source.skeleton.signature(), first_target.signature())

    def test_visual_is_valid_bilingual_svg(self) -> None:
        results = [
            {
                "mode": "same_character",
                "instances": 1,
                "sample_ms_p50": 0.001,
                "sample_ms_p95": 0.002,
                "sample_ms_p99": 0.003,
                "tick_ms_p95": 0.002,
            }
        ] + [
            {
                "mode": "different_character",
                "instances": instances,
                "sample_ms_p50": 0.003,
                "sample_ms_p95": 0.004,
                "sample_ms_p99": 0.005,
                "tick_ms_p95": 0.004 * instances,
            }
            for instances in (1, 10, 50, 100)
        ]
        report = {
            "passed": True,
            "method": {"iterations": 2000},
            "results": results,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.svg"
            write_visual(path, report)
            ET.parse(path)
            content = path.read_text(encoding="utf-8")
        self.assertIn("Native Runtime Sampling", content)
        self.assertIn("原生运行时采样", content)
        self.assertIn("16.667 ms", content)


if __name__ == "__main__":
    unittest.main()
