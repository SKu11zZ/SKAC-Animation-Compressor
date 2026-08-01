from __future__ import annotations

import unittest

import numpy as np

from skac_codec.math3d import (
    euler_order_to_quaternion,
    matrix_to_quaternion,
    quaternion_angular_error_degrees,
    quaternion_to_matrix,
    quaternion_to_euler_order,
    quaternion_slerp,
)


class CodecMathTests(unittest.TestCase):
    def test_matrix_quaternion_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        angles = rng.uniform(-2.0, 2.0, size=(100, 3))
        source = euler_order_to_quaternion("ZXY", angles)
        restored = matrix_to_quaternion(quaternion_to_matrix(source))
        self.assertLess(
            float(np.max(quaternion_angular_error_degrees(source, restored))), 1e-5
        )

    def test_scalar_slerp(self) -> None:
        start = euler_order_to_quaternion("XYZ", np.asarray([0.0, 0.0, 0.0]))
        end = euler_order_to_quaternion("XYZ", np.asarray([0.0, 0.0, np.pi]))
        middle = quaternion_slerp(start, end, 0.5)
        expected = euler_order_to_quaternion("XYZ", np.asarray([0.0, 0.0, np.pi / 2.0]))
        self.assertLess(
            float(quaternion_angular_error_degrees(middle, expected)), 1e-5
        )

    def test_all_bvh_euler_orders_round_trip(self) -> None:
        rng = np.random.default_rng(20260801)
        for order in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
            angles = rng.uniform(-1.2, 1.2, size=(100, 3))
            source = euler_order_to_quaternion(order, angles)
            restored_angles = quaternion_to_euler_order(source, order)
            restored = euler_order_to_quaternion(order, restored_angles)
            error = quaternion_angular_error_degrees(source, restored)
            self.assertLess(float(np.max(error)), 1e-5, order)


if __name__ == "__main__":
    unittest.main()
