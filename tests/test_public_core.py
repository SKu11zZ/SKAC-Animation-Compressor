from __future__ import annotations

import unittest

import numpy as np

from skac_public_core import Motion, Skeleton, build_profile, retarget_motion


class PublicCoreTests(unittest.TestCase):
    def test_cross_skeleton_retarget_maps_common_joints(self) -> None:
        source = Skeleton(
            names=("src:Hips", "src:Spine", "src:Head"),
            parents=np.array([-1, 0, 1]),
            offsets=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]),
            height=2.0,
        )
        target = Skeleton(
            names=("Hips", "Spine", "SpineExtra", "Head"),
            parents=np.array([-1, 0, 1, 2]),
            offsets=np.array([[0, 0, 0], [0, 0.5, 0], [0, 0.5, 0], [0, 1, 0]]),
            height=1.0,
        )
        source_rotations = np.zeros((2, 3, 4))
        source_rotations[..., 0] = 1.0
        source_rotations[:, 1] = np.array([0.0, 1.0, 0.0, 0.0])
        motion = Motion(source_rotations, np.array([[2, 4, 6], [4, 6, 8]]))

        profile = build_profile(source, target)
        result = retarget_motion(profile, motion)

        self.assertEqual(profile.target_to_source, ((0, 0), (1, 1), (3, 2)))
        np.testing.assert_allclose(result.local_rotations[:, 1], source_rotations[:, 1])
        np.testing.assert_allclose(result.local_rotations[:, 2], [[1, 0, 0, 0]] * 2)
        np.testing.assert_allclose(result.root_positions, motion.root_positions * 0.5)

    def test_profile_rejects_animation_specific_adjustment(self) -> None:
        from skac_public_core import RetargetProfile

        with self.assertRaises(ValueError):
            RetargetProfile(
                source_joint_count=1,
                target_joint_count=1,
                target_to_source=((0, 0),),
                root_translation_scale=1.0,
                per_animation_adjustment=True,
            )


if __name__ == "__main__":
    unittest.main()
