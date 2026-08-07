from __future__ import annotations

import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import skac_codec.fbx as fbx


class FbxAdapterTests(unittest.TestCase):
    def test_blender_bridge_script_is_valid_python(self) -> None:
        source = fbx._bridge_script().read_text(encoding="utf-8")
        ast.parse(source)

    def test_missing_explicit_blender_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-blender"
            with self.assertRaises(fbx.FbxBackendUnavailable):
                fbx.find_blender(missing)

    def test_bridge_uses_argument_array_and_no_shell(self) -> None:
        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertIs(kwargs["shell"], False)
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "replace")
            self.assertIn("--factory-startup", command)
            report_path = Path(command[command.index("--report") + 1])
            report_path.write_text(
                json.dumps({"status": "ok", "armature_count": 1}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch("skac_codec.fbx.subprocess.run", side_effect=fake_run):
            report = fbx._run_bridge(
                Path("blender"), ["validate", "--input", "character.fbx"], timeout_seconds=5
            )
        self.assertEqual(report["armature_count"], 1)

    def test_extract_and_inject_replace_only_completed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blender = root / "blender"
            source = root / "source.fbx"
            template = root / "target.fbx"
            animation = root / "target-animation.bvh"
            extracted = root / "source.bvh"
            output = root / "animated.fbx"
            for path in (blender, source, template, animation):
                path.write_bytes(b"fixture")

            def fake_bridge(
                _blender: Path, arguments: list[str], *, timeout_seconds: int
            ) -> dict[str, object]:
                del timeout_seconds
                destination = Path(arguments[arguments.index("--output") + 1])
                destination.write_bytes(b"completed-output")
                return {"status": "ok", "armature_count": 1}

            with mock.patch("skac_codec.fbx._run_bridge", side_effect=fake_bridge):
                fbx.extract_fbx_to_bvh(
                    source, extracted, blender_executable=blender
                )
                fbx.inject_bvh_into_fbx(
                    template,
                    animation,
                    output,
                    blender_executable=blender,
                )
            self.assertEqual(extracted.read_bytes(), b"completed-output")
            self.assertEqual(output.read_bytes(), b"completed-output")


if __name__ == "__main__":
    unittest.main()
