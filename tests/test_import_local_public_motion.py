from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.import_local_public_motion import import_cache


class ImportLocalPublicMotionTests(unittest.TestCase):
    def test_import_uses_neutral_names_and_path_free_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "mixamo": root / "source-a",
                "manny": root / "source-b",
                "smpl": root / "source-c",
            }
            for source in sources.values():
                source.mkdir()
            (sources["mixamo"] / "descriptive motion.fbx").write_bytes(b"mixamo")
            (sources["manny"] / "descriptive motion.fbx").write_bytes(b"manny")
            (sources["smpl"] / "descriptive motion.npz").write_bytes(b"smpl")
            (sources["smpl"] / "license.txt").write_text("public", encoding="utf-8")
            output = root / "cache"
            report = import_cache(sources, output)
            manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)

        self.assertEqual(report, manifest)
        self.assertEqual(
            [item["file_count"] for item in manifest["families"]], [1, 1, 2]
        )
        self.assertNotIn("descriptive motion", manifest_text)
        self.assertNotIn(str(root), manifest_text)
        self.assertTrue(
            all(
                entry["relative_path"].startswith(f'{family["family"]}/')
                for family in manifest["families"]
                for entry in family["files"]
            )
        )

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cache"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                import_cache(
                    {"mixamo": root, "manny": root, "smpl": root}, output
                )


if __name__ == "__main__":
    unittest.main()
