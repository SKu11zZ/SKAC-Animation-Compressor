from __future__ import annotations

import unittest

import numpy as np

from skac_codec.bvh import dumps_bvh, loads_bvh
from skac_codec.math3d import quaternion_angular_error_degrees


SAMPLE_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Chest
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 8 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333333
0 0 0 0 0 0 0 0 0
1 0.5 0 10 5 -2 20 -3 4
2 1 0 20 10 -4 40 -6 8
"""


class BvhIoTests(unittest.TestCase):
    def test_bvh_read_write_preserves_motion(self) -> None:
        source = loads_bvh(SAMPLE_BVH)
        restored = loads_bvh(dumps_bvh(source))
        self.assertEqual(source.skeleton.signature(), restored.skeleton.signature())
        self.assertTrue(np.allclose(source.local_translations, restored.local_translations, atol=1e-8))
        error = quaternion_angular_error_degrees(
            source.local_rotations, restored.local_rotations
        )
        self.assertLess(float(np.max(error)), 1e-5)


if __name__ == "__main__":
    unittest.main()
