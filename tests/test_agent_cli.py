from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from skac_codec.agent import PROTOCOL, execute_request, main


SINGLE_JOINT_BVH = """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  End Site
  {
    OFFSET 0 1 0
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333333
0 0 0 0 0 0
1 0 0 5 0 0
2 0 0 10 0 0
"""


def request(request_id: str, operation: str, **arguments: object) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "operation": operation,
        "arguments": arguments,
    }


class AgentCliTests(unittest.TestCase):
    def test_capabilities_are_machine_readable(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["capabilities"]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["protocol"], PROTOCOL)
        self.assertIn("encode", report["operations"])
        self.assertIn("quality_gate_same", report["operations"])
        self.assertIn("quality_gate_different", report["operations"])
        self.assertIn("runtime_skeleton", report["operations"])
        self.assertNotIn(
            "target", report["operations"]["quality_gate_same"]["optional"]
        )
        self.assertIn(
            "target", report["operations"]["quality_gate_different"]["required"]
        )

    def test_encode_inspect_decode_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "motions").mkdir()
            (workspace / "motions" / "source.bvh").write_text(
                SINGLE_JOINT_BVH, encoding="utf-8"
            )

            encoded, status = execute_request(
                request(
                    "e1",
                    "encode",
                    input="motions/source.bvh",
                    output="artifacts/motion.skac",
                    quality="high",
                ),
                workspace,
            )
            self.assertEqual(status, 0)
            self.assertTrue(encoded["ok"])
            self.assertEqual(encoded["result"]["artifact"]["path"], "artifacts/motion.skac")

            inspected, status = execute_request(
                request("i1", "inspect", input="artifacts/motion.skac"), workspace
            )
            self.assertEqual(status, 0)
            self.assertEqual(inspected["result"]["format_major"], 1)

            profiled, status = execute_request(
                request(
                    "p1",
                    "profile",
                    source="artifacts/motion.skac",
                    target="motions/source.bvh",
                    output="profiles/target.json",
                ),
                workspace,
            )
            self.assertEqual(status, 0)
            self.assertEqual(profiled["result"]["mapped_joint_count"], 1)

            runtime_target, status = execute_request(
                request(
                    "r1",
                    "runtime_skeleton",
                    target="motions/source.bvh",
                    output="profiles/target.runtime.json",
                ),
                workspace,
            )
            self.assertEqual(status, 0)
            self.assertEqual(runtime_target["result"]["joint_count"], 1)
            self.assertTrue((workspace / "profiles" / "target.runtime.json").is_file())

            decoded, status = execute_request(
                request(
                    "d1",
                    "decode",
                    input="artifacts/motion.skac",
                    target="motions/source.bvh",
                    profile="profiles/target.json",
                    output="results/restored.bvh",
                ),
                workspace,
            )
            self.assertEqual(status, 0)
            self.assertTrue(decoded["result"]["retargeted"])
            self.assertTrue((workspace / "results" / "restored.bvh").is_file())

    def test_rejects_path_escape_unknown_fields_and_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            escaped, status = execute_request(
                request("bad-path", "inspect", input="../motion.skac"), workspace
            )
            self.assertEqual(status, 2)
            self.assertEqual(escaped["error"]["code"], "invalid_request")

            unknown, status = execute_request(
                request("bad-field", "inspect", input="motion.skac", typo=True), workspace
            )
            self.assertEqual(status, 2)
            self.assertIn("unknown fields", unknown["error"]["message"])

            (workspace / "source.bvh").write_text(SINGLE_JOINT_BVH, encoding="utf-8")
            (workspace / "motion.skac").write_bytes(b"keep")
            existing, status = execute_request(
                request("existing", "encode", input="source.bvh", output="motion.skac"),
                workspace,
            )
            self.assertEqual(status, 2)
            self.assertEqual((workspace / "motion.skac").read_bytes(), b"keep")

    def test_jsonl_continues_after_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "source.bvh").write_text(SINGLE_JOINT_BVH, encoding="utf-8")
            payload = "\n".join(
                [
                    json.dumps(request("bad", "missing")),
                    json.dumps(
                        request(
                            "good",
                            "encode",
                            input="source.bvh",
                            output="motion.skac",
                        )
                    ),
                ]
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                original = __import__("sys").stdin
                try:
                    __import__("sys").stdin = io.TextIOWrapper(
                        io.BytesIO(payload.encode("utf-8")), encoding="utf-8"
                    )
                    self.assertEqual(
                        main(["run", "--workspace", str(workspace), "--jsonl"]), 2
                    )
                finally:
                    __import__("sys").stdin = original
            responses = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([item["request_id"] for item in responses], ["bad", "good"])
            self.assertFalse(responses[0]["ok"])
            self.assertTrue(responses[1]["ok"])
            self.assertTrue((workspace / "motion.skac").is_file())

    def test_same_character_quality_gate_has_its_own_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "source.bvh").write_text(SINGLE_JOINT_BVH, encoding="utf-8")
            response, status = execute_request(
                request(
                    "gate",
                    "quality_gate_same",
                    source="source.bvh",
                    report="reports/gate.json",
                    visual="reports/gate.svg",
                    decode_iterations=1,
                ),
                workspace,
            )
            self.assertEqual(status, 0)
            self.assertTrue(response["ok"])
            self.assertTrue(response["result"]["passed"])
            self.assertEqual(
                response["result"]["evaluation_case"], "same_character"
            )
            self.assertTrue((workspace / "reports" / "gate.json").is_file())
            self.assertTrue((workspace / "reports" / "gate.svg").is_file())

    def test_same_character_operation_rejects_target_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "source.bvh").write_text(SINGLE_JOINT_BVH, encoding="utf-8")
            response, status = execute_request(
                request(
                    "gate",
                    "quality_gate_same",
                    source="source.bvh",
                    target="source.bvh",
                    report="gate.json",
                    visual="gate.svg",
                ),
                workspace,
            )
            self.assertEqual(status, 2)
            self.assertFalse(response["ok"])
            self.assertIn("unknown fields: target", response["error"]["message"])

    def test_different_character_quality_gate_requires_and_uses_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "source.bvh").write_text(SINGLE_JOINT_BVH, encoding="utf-8")
            response, status = execute_request(
                request(
                    "gate",
                    "quality_gate_different",
                    source="source.bvh",
                    target="source.bvh",
                    report="gate.json",
                    visual="gate.svg",
                    minimum_core_coverage=0.0,
                    decode_iterations=1,
                    pipeline_iterations=1,
                    frame_samples=10,
                ),
                workspace,
            )
            self.assertEqual(status, 0)
            self.assertTrue(response["ok"])
            self.assertEqual(
                response["result"]["evaluation_case"], "different_character"
            )


if __name__ == "__main__":
    unittest.main()
