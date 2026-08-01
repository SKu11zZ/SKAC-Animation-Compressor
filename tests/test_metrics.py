import unittest

import numpy as np

from skac_benchmark.metrics import evaluate_motion, penetration_percent


class MetricsTests(unittest.TestCase):
    def test_global_and_root_aligned_mse_are_separate(self) -> None:
        truth = np.zeros((2, 22, 3), dtype=np.float64)
        predicted = truth + np.array([2.0, 0.0, 0.0])
        metrics = evaluate_motion(predicted, truth, character_height=2.0, root_index=0)
        self.assertAlmostEqual(metrics.global_mse_raw, 1.0 / 3.0)
        self.assertAlmostEqual(metrics.local_mse_raw, 0.0)
        self.assertAlmostEqual(metrics.san_paper_position_error_x1e3, 1000.0)

    def test_geometry_metrics(self) -> None:
        positions = np.zeros((2, 22, 3), dtype=np.float64)
        metrics = evaluate_motion(
            positions,
            positions,
            character_height=1.0,
            root_index=0,
            penetrated_vertices=np.array([1, 2]),
            total_limb_vertices=np.array([10, 20]),
            contact_distances_cm=np.array([[1.0, 3.0], [2.0, 4.0]]),
        )
        self.assertAlmostEqual(metrics.penetration_percent, 10.0)
        self.assertAlmostEqual(metrics.contact_distance_cm, 2.5)

    def test_invalid_penetration_counts_fail(self) -> None:
        with self.assertRaises(ValueError):
            penetration_percent([2], [1])


if __name__ == "__main__":
    unittest.main()
