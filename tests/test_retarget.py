from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np

from skac_codec.format import PREFIX, decode_bytes, encode_bytes
from skac_codec.format_v2 import encode_v2_bytes
from skac_codec.math3d import euler_order_to_quaternion, quaternion_angular_error_degrees
from skac_codec.model import MotionClip, Skeleton
from skac_codec.retarget import (
    build_retarget_profile,
    compile_retarget_profile,
    load_retarget_profile,
    retarget_motion,
    retarget_motion_reference,
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
    def test_v2_bitstream_is_independent_of_target_profiles(self) -> None:
        source = source_clip()
        encoded = encode_v2_bytes(source)
        encoded_sha256 = hashlib.sha256(encoded).hexdigest()
        decoded = decode_bytes(encoded)
        targets = (target_skeleton(1.2, True), target_skeleton(1.5, False))
        profiles = [
            build_retarget_profile(decoded.skeleton, target, contact_lock=False)
            for target in targets
        ]
        for target, profile in zip(targets, profiles, strict=True):
            result, _ = retarget_motion(decoded, target, profile)
            self.assertEqual(result.frame_count, source.frame_count)

        self.assertEqual(hashlib.sha256(encoded).hexdigest(), encoded_sha256)
        self.assertEqual(encode_v2_bytes(source), encoded)
        fields = PREFIX.unpack_from(encoded)
        metadata_start = PREFIX.size
        metadata_end = metadata_start + int(fields[4])
        metadata_text = encoded[metadata_start:metadata_end].decode("utf-8")
        metadata = json.loads(metadata_text)
        self.assertEqual(metadata["skeleton_sha256"], source.skeleton.signature())
        for target, profile in zip(targets, profiles, strict=True):
            self.assertNotIn(target.signature(), metadata_text)
            self.assertNotIn(profile.signature(), metadata_text)

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

    def test_compiled_frame_runtime_matches_matrix_reference(self) -> None:
        source = source_clip()
        target = target_skeleton(1.2, True)
        profile = build_retarget_profile(source.skeleton, target)
        runtime = compile_retarget_profile(source.skeleton, target, profile)
        compiled, diagnostics = retarget_motion(
            source, target, profile, runtime=runtime
        )
        reference, _ = retarget_motion_reference(source, target, profile)
        error = quaternion_angular_error_degrees(
            compiled.local_rotations, reference.local_rotations
        )
        self.assertLessEqual(float(np.max(error)), 1e-8)
        self.assertTrue(
            np.allclose(compiled.local_translations, reference.local_translations)
        )
        self.assertEqual(diagnostics["runtime_mode"], "compiled_quaternion_frame_v2")
        self.assertEqual(profile.core_coverage, 1.0)

    def test_profile_v2_disambiguates_smpl_shoulder_chain(self) -> None:
        channels = (("Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"),) + (
            ("Zrotation", "Xrotation", "Yrotation"),
        ) * 3
        source = Skeleton(
            names=("Hips", "Spine", "LeftShoulder", "LeftArm"),
            parents=np.asarray([-1, 0, 1, 2], dtype=np.int32),
            offsets=np.asarray([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float64),
            channels=channels,
            end_site_offsets=np.zeros((4, 3), dtype=np.float64),
            has_end_sites=np.zeros(4, dtype=np.bool_),
        )
        target = Skeleton(
            names=("pelvis", "spine1", "left_collar", "left_shoulder"),
            parents=np.asarray([-1, 0, 1, 2], dtype=np.int32),
            offsets=np.asarray([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float64),
            channels=channels,
            end_site_offsets=np.zeros((4, 3), dtype=np.float64),
            has_end_sites=np.zeros(4, dtype=np.bool_),
        )
        profile = build_retarget_profile(source, target)
        mapping = {item.target_name: item.source_name for item in profile.transfers}
        self.assertEqual(mapping["left_collar"], "LeftShoulder")
        self.assertEqual(mapping["left_shoulder"], "LeftArm")

    def test_legacy_profile_can_still_be_loaded(self) -> None:
        profile = build_retarget_profile(source_clip().skeleton, target_skeleton(1.2, True))
        legacy = profile.to_dict(include_hash=False)
        legacy["schema_version"] = "1.0.0"
        legacy.pop("builder")
        legacy.pop("runtime_plan")
        for transfer in legacy["transfers"]:
            transfer.pop("basis_quaternion")
            transfer.pop("mapping_method")
        canonical = json.dumps(
            legacy, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        legacy["profile_sha256"] = hashlib.sha256(canonical).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            restored = load_retarget_profile(path)
        self.assertEqual(restored.schema_version, "1.0.0")
        runtime = compile_retarget_profile(
            source_clip().skeleton, target_skeleton(1.2, True), restored
        )
        self.assertGreater(len(runtime.target_evaluation_order), 1)


if __name__ == "__main__":
    unittest.main()
