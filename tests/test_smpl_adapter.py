from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skac_codec.cli import main
from skac_codec.format import decode_bytes
from skac_codec.math3d import quaternion_angular_error_degrees
from skac_codec.smpl import read_smpl_npz, smplx55_proxy_skeleton


class SmplAdapterTests(unittest.TestCase):
    def _write_motion(self, path: Path) -> None:
        poses = np.zeros((4, 165), dtype=np.float64)
        poses[:, 2] = np.linspace(0.0, 0.3, 4)
        trans = np.zeros((4, 3), dtype=np.float64)
        trans[:, 0] = np.linspace(0.0, 0.5, 4)
        np.savez(path, poses=poses, trans=trans, mocap_frame_rate=np.asarray(120.0))

    def test_numeric_npz_loads_without_body_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.npz"
            self._write_motion(path)
            clip = read_smpl_npz(path)
        self.assertEqual(clip.frame_count, 4)
        self.assertEqual(clip.skeleton.joint_count, 55)
        self.assertAlmostEqual(clip.frame_time, 1.0 / 120.0)
        self.assertAlmostEqual(float(clip.local_translations[-1, 0, 0]), 0.5)

    def test_encode_cli_accepts_smpl_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "motion.npz"
            output = root / "motion.skac"
            self._write_motion(source)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["encode", str(source), "-o", str(output)]), 0)
            report = json.loads(stdout.getvalue())
            decoded = decode_bytes(output.read_bytes())
            source_clip = read_smpl_npz(source)
        self.assertEqual(report["input_format"], "smplx_parameter_stream")
        self.assertEqual(decoded.skeleton.signature(), smplx55_proxy_skeleton().signature())
        self.assertLess(
            float(
                np.max(
                    quaternion_angular_error_degrees(
                        decoded.local_rotations, source_clip.local_rotations
                    )
                )
            ),
            0.1,
        )

    def test_non_motion_npz_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shape-only.npz"
            np.savez(path, betas=np.zeros(16))
            with self.assertRaisesRegex(ValueError, "missing numeric animation"):
                read_smpl_npz(path)


if __name__ == "__main__":
    unittest.main()
