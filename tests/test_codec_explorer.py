from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.build_codec_explorer import build_explorer_data, write_html_bundle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "codec_showcase_8x20_public.json"
HTML = ROOT / "reports" / "codec_explorer.html"


class CodecExplorerTests(unittest.TestCase):
    def test_browser_bundle_is_complete_and_reproducible(self) -> None:
        source_bytes = REPORT.read_bytes()
        report = json.loads(source_bytes.decode("utf-8"))
        data = build_explorer_data(report, source_bytes)
        self.assertEqual(len(data["animations"]), 20)
        self.assertEqual(len(data["characters"]), 8)
        self.assertEqual(len(data["samples"]), 160)
        self.assertEqual(
            len(
                {
                    (sample["animation"], sample["character"])
                    for sample in data["samples"]
                }
            ),
            160,
        )
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "codec_explorer.html"
            shutil.copyfile(HTML, generated)
            write_html_bundle(generated, data)
            self.assertEqual(generated.read_bytes(), HTML.read_bytes())

    def test_html_is_offline_and_exposes_player_and_metrics(self) -> None:
        html = HTML.read_text(encoding="utf-8")
        folded = html.casefold()
        self.assertIn('id="animationselect"', folded)
        self.assertIn('id="skeletoncanvas"', folded)
        self.assertIn('id="metricsTitle"'.casefold(), folded)
        self.assertIn('accept=".bvh"', folded)
        self.assertIn('id="codecexplorerdata"', folded)
        self.assertNotIn("https://", folded)
        self.assertNotIn("http://", folded)
        self.assertNotIn("<iframe", folded)
        match = re.search(
            r'<script id="codecExplorerData" type="application/json">(.*?)</script>',
            html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        embedded = json.loads(match.group(1))
        self.assertEqual(len(embedded["samples"]), 160)


if __name__ == "__main__":
    unittest.main()
