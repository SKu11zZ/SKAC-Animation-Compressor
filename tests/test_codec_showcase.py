from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from skac_codec.format import CodecSettings
from tests.test_codec_cli import SINGLE_JOINT_BVH
from tools.run_codec_showcase import run_benchmark, select_animations, write_report, write_visual


class CodecShowcaseTests(unittest.TestCase):
    def test_selection_uses_the_common_animation_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for character in ("One", "Two"):
                character_root = root / character
                character_root.mkdir()
                for animation in ("A", "B", "C"):
                    (character_root / f"{animation}.bvh").write_text("", encoding="utf-8")
            (root / "Two" / "C.bvh").unlink()
            first = select_animations(
                root, characters=("One", "Two"), count=2, seed=17
            )
            second = select_animations(
                root, characters=("One", "Two"), count=2, seed=17
            )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"A", "B"})

    def test_tiny_roundtrip_writes_safe_json_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for character in ("One", "Two"):
                character_root = root / character
                character_root.mkdir()
                (character_root / "Move.bvh").write_text(
                    SINGLE_JOINT_BVH, encoding="utf-8"
                )
            report = run_benchmark(
                root,
                characters=("One", "Two"),
                animations_per_character=1,
                seed=17,
                decode_iterations=1,
                settings=CodecSettings.preset("high"),
            )
            json_path = root / "report.json"
            svg_path = root / "report.svg"
            write_report(json_path, report)
            write_visual(svg_path, report)
            restored = json.loads(json_path.read_text(encoding="utf-8"))
            ElementTree.parse(svg_path)
            svg = svg_path.read_text(encoding="utf-8").casefold()
        self.assertTrue(restored["passed"])
        self.assertEqual(restored["sampling"]["sample_count"], 2)
        self.assertIn("mib", svg)
        self.assertIn("decode · milliseconds", svg)
        self.assertNotIn("realtime", svg)
        self.assertNotIn("<script", svg)


if __name__ == "__main__":
    unittest.main()
