from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.audit_release import audit


class ReleaseAuditTests(unittest.TestCase):
    def test_accepts_small_public_text_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("public benchmark\n", encoding="utf-8")
            self.assertEqual(audit(root, []), [])

    def test_rejects_release_excluded_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            (reports / "prediction.bvh").write_bytes(b"motion payload")
            self.assertIn(
                "release-excluded file type: reports/prediction.bvh",
                audit(root, []),
            )

    def test_rejects_absolute_machine_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine_path = "C:" + "\\" + "Users" + "\\" + "example" + "\\" + "cache"
            (root / "README.md").write_text(
                f"local cache: {machine_path}\n", encoding="utf-8"
            )
            self.assertIn(
                "absolute filesystem path in file: README.md",
                audit(root, []),
            )

    def test_rejects_out_of_band_deny_term(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("restricted-token\n", encoding="utf-8")
            self.assertIn(
                "denylisted term in file: README.md",
                audit(root, ["restricted-token"]),
            )

    def test_accepts_static_svg_and_rejects_active_svg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            report = reports / "report.svg"
            report.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>',
                encoding="utf-8",
            )
            self.assertEqual(audit(root, []), [])
            report.write_text("<svg><script>alert(1)</script></svg>", encoding="utf-8")
            self.assertIn(
                "unsafe active or external SVG content: reports/report.svg",
                audit(root, []),
            )


if __name__ == "__main__":
    unittest.main()
