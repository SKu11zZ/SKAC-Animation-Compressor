from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skac_codec.format import decode_bytes, encode_bytes
from skac_codec.math3d import euler_order_to_quaternion, quaternion_angular_error_degrees
from skac_codec.model import MotionClip, Skeleton
from skac_codec.retarget import (
    build_retarget_profile,
    load_retarget_profile,
    retarget_motion,
    save_retarget_profile,
)


def source_clip() -> MotionClip:
    names = (
        "Hips", "Spine", "LeftUpLeg", "LeftLeg", "LeftFoot",
        "RightUpLeg", "RightLeg", "RightFoot",
    )
    parents = np.asarray([-1, 0, 0, 2, 3, 0, 5, 6], dtype=np.int32)
    offsets = np.asarray(
        [
            [0, 10, 0], [0, 10, 0], [4, -8, 0], [0, -10, 0], [0, -9, 2],
            [-4, -8, 0], [0, -10, 0], [0, -9, 2],
        ],
        dtype=np.float64,
    )
    rotation_channels = ("Zrotation", "Xrotation", "Yrotation")
    skeleton = Skeleton(
        names=names,
        parents=parents,
        offsets=offsets,
        channels=(("Xposition", "Yposition", "Zposition") + rotation_channels,) + (rotation_channels,) * 7,
        end_site_offsets=np.zeros((8, 3), dtype=np.float64),
        has_end_sites=np.zeros(8, dtype=np.bool_),
    )
    frames = 30
    phase = np.linspace(0.0, np.pi, frames)
    rotations = np.zeros((frames, 8, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    rotations[:, 0] = euler_order_to_quaternion(
        "ZXY", np.stack((np.zeros(frames), np.zeros(frames), 0.3 * np.sin(phase)), axis=1)
    )
    rotations[:, 2] = euler_order_to_quaternion(
        "ZXY", np.stack((0.25 * np.sin(phase), np.zeros(frames), np.zeros(frames)), axis=1)
    )
    rotations[:, 5] = euler_order_to_quaternion(
        "ZXY", np.stack((-0.25 * np.sin(phase), np.zeros(frames), np.zeros(frames)), axis=1)
    )
    translations = np.broadcast_to(offsets, (frames, 8, 3)).copy()
    translations[:, 0, 0] += np.linspace(0.0, 20.0, frames)
    return MotionClip(skeleton, rotations, translations, 1.0 / 30.0)


def target_skeleton(scale: float, engine_names: bool) -> Skeleton:
    rotation_channels = ("Zrotation", "Xrotation", "Yrotation")
    if engine_names:
        names = (
            "root", "pelvis", "spine_01", "thigh_l", "calf_l", "foot_l",
            "thigh_r", "calf_r", "foot_r",
        )
        parents = np.asarray([-1, 0, 1, 1, 3, 4, 1, 6, 7], dtype=np.int32)
        base_offsets = np.asarray(
            [
                [0, 0, 0], [0, 10, 0], [0, 10, 0], [4, -8, 0], [0, -10, 0],
                [0, -9, 2], [-4, -8, 0], [0, -10, 0], [0, -9, 2],
            ],
            dtype=np.float64,
        )
        channels = (("Xposition", "Yposition", "Zposition") + rotation_channels,) + (rotation_channels,) * 8
    else:
        names = source_clip().skeleton.names
        parents = source_clip().skeleton.parents
        base_offsets = source_clip().skeleton.offsets
        channels = source_clip().skeleton.channels
    return Skeleton(
        names=names,
        parents=parents,
        offsets=base_offsets * scale,
        channels=channels,
        end_site_offsets=np.zeros((len(names), 3), dtype=np.float64),
        has_end_sites=np.zeros(len(names), dtype=np.bool_),
    )


class RetargetTests(unittest.TestCase):
    def test_one_encoded_clip_targets_two_skeletons(self) -> None:
        source = source_clip()
        decoded = decode_bytes(encode_bytes(source))
        first_skeleton = target_skeleton(1.2, True)
        second_skeleton = target_skeleton(1.5, False)
        first_profile = build_retarget_profile(
            decoded.skeleton, first_skeleton, contact_lock=False
        )
        second_profile = build_retarget_profile(
            decoded.skeleton, second_skeleton, contact_lock=False
        )
        first, first_info = retarget_motion(decoded, first_skeleton, first_profile)
        second, second_info = retarget_motion(decoded, second_skeleton, second_profile)

        self.assertNotEqual(first.skeleton.signature(), second.skeleton.signature())
        self.assertEqual(first.frame_count, source.frame_count)
        self.assertEqual(second.frame_count, source.frame_count)
        self.assertGreater(first_info["mapped_joint_count"], 4)
        self.assertGreater(second_info["mapped_joint_count"], 4)
        self.assertTrue(np.allclose(first.local_rotations[:, 0, 0], 1.0))
        pelvis_error = quaternion_angular_error_degrees(
            first.local_rotations[:, 1], np.asarray([1.0, 0.0, 0.0, 0.0])
        )
        self.assertGreater(float(np.max(pelvis_error)), 1.0)
        self.assertTrue(
            np.allclose(
                second.local_translations[:, 0, 0] - second.skeleton.offsets[0, 0],
                (decoded.local_translations[:, 0, 0] - decoded.skeleton.offsets[0, 0])
                * second_profile.root_translation_scale,
            )
        )

    def test_profile_round_trip_is_hash_pinned(self) -> None:
        source = source_clip().skeleton
        target = target_skeleton(1.2, True)
        profile = build_retarget_profile(source, target)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.profile.json"
            save_retarget_profile(path, profile)
            restored = load_retarget_profile(path)
        self.assertEqual(restored.signature(), profile.signature())
        self.assertEqual(restored.source_skeleton_sha256, source.signature())
        self.assertEqual(restored.target_skeleton_sha256, target.signature())

    def test_profile_rejects_a_different_target_skeleton(self) -> None:
        source = source_clip()
        first_target = target_skeleton(1.2, True)
        second_target = target_skeleton(1.5, False)
        profile = build_retarget_profile(source.skeleton, first_target)
        with self.assertRaisesRegex(ValueError, "target skeleton"):
            retarget_motion(source, second_target, profile)


if __name__ == "__main__":
    unittest.main()
