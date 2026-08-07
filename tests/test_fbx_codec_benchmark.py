from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from tools.run_fbx_codec_benchmark import _aggregate, _inventory, _svg


class FbxCodecBenchmarkTests(unittest.TestCase):
    def test_inventory_uses_only_neutral_family_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ("mixamo", "manny"):
                (root / family).mkdir()
                (root / family / f"{family}_0001.fbx").write_bytes(b"fixture")
            records = _inventory(root)
        self.assertEqual(
            [(family, identifier) for family, identifier, _ in records],
            [("mixamo", "mixamo_0001"), ("manny", "manny_0001")],
        )

    def test_aggregate_and_visual_are_concrete_static_svg(self) -> None:
        def sample(family: str) -> dict[str, object]:
            return {
                "family": family, "source_frame_count": 10, "frame_count": 10,
                "playback_seconds": 1.0, "joint_count": 55, "encode_clip_ms": 100.0,
                "decode_clip_ms_median": 10.0, "raw_channel_bytes_float32": 1000,
                "encoded_bytes": 200, "rotation_error_degrees_mean": 0.01,
                "rotation_error_degrees_max": 0.02, "root_translation_error_max": 0.001,
                "global_position_error_max": 0.002,
                "global_position_error_max_height_fraction": 0.0002,
                "root_translation_error_max_height_fraction": 0.0001,
            }
        samples = [sample("mixamo"), sample("manny")]
        aggregate = _aggregate(samples)
        self.assertEqual(aggregate["compression_ratio_vs_float32_channels"], 5.0)
        report = {
            "aggregate": aggregate,
            "families": [
                {"family": family, **_aggregate([item])}
                for family, item in zip(("mixamo", "manny"), samples)
            ],
            "backend": {"blender_version": "Blender 4.5"},
            "environment": {"python": "3.12"},
            "passed": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.svg"
            path.write_text(_svg(report), encoding="utf-8")
            ElementTree.parse(path)
            text = path.read_text(encoding="utf-8").casefold()
        self.assertNotIn("<script", text)
        self.assertIn("mixamo", text)
        self.assertIn("manny", text)
        self.assertIn("quality gate: fail", text)


if __name__ == "__main__":
    unittest.main()
