from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.run_smpl_codec_benchmark import _aggregate, _motion_inventory


class SmplCodecBenchmarkTests(unittest.TestCase):
    def test_inventory_separates_motion_and_shape_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(
                root / "smpl_0001.npz",
                poses=np.zeros((2, 165)),
                trans=np.zeros((2, 3)),
                mocap_frame_rate=np.asarray(120.0),
            )
            np.savez(root / "smpl_0002.npz", betas=np.zeros(16))
            motions, skipped = _motion_inventory(root)
        self.assertEqual([(item[0], item[2]) for item in motions], [("smpl_0001", 2)])
        self.assertEqual(skipped, ["smpl_0002"])

    def test_aggregate_uses_concrete_byte_totals(self) -> None:
        sample = {
            "frame_count": 10, "joint_count": 55, "duration_seconds": 0.075,
            "encode_clip_ms": 30.0, "decode_clip_ms": 1.0,
            "raw_channel_bytes_float32": 6720, "encoded_bytes": 840,
            "rotation_error_degrees_mean": 0.01, "rotation_error_degrees_max": 0.02,
            "root_translation_error_mean": 0.001, "root_translation_error_max": 0.002,
            "global_position_error_max": 0.003,
        }
        aggregate = _aggregate([sample])
        self.assertEqual(aggregate["raw_channel_bytes_float32"], 6720)
        self.assertEqual(aggregate["encoded_bytes"], 840)
        self.assertEqual(aggregate["compression_ratio_vs_float32_channels"], 8.0)


if __name__ == "__main__":
    unittest.main()
