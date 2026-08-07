from __future__ import annotations

import unittest

import numpy as np

from skac_codec.format import (
    CodecSettings,
    SkacFormatError,
    _decode_rotations,
    _encode_rotations,
    _quaternion_slerp_unit,
    _rotation_key_indices,
    _rotation_key_indices_multi,
    decode_bytes,
    encode_bytes,
    quantize_rotation_samples,
    quantize_translation_samples,
)
from skac_codec.math3d import (
    euler_order_to_quaternion,
    normalize_quaternions,
    quaternion_slerp,
)
from skac_codec.metrics import roundtrip_metrics
from skac_codec.model import MotionClip, Skeleton


def sample_clip(frame_count: int = 120) -> MotionClip:
    skeleton = Skeleton(
        names=("Root", "Chest", "Hand"),
        parents=np.asarray([-1, 0, 1], dtype=np.int32),
        offsets=np.asarray([[0, 0, 0], [0, 10, 0], [5, 8, 0]], dtype=np.float64),
        channels=(
            ("Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"),
            ("Zrotation", "Xrotation", "Yrotation"),
            ("Zrotation", "Xrotation", "Yrotation"),
        ),
        end_site_offsets=np.asarray([[0, 0, 0], [0, 0, 0], [4, 0, 0]], dtype=np.float64),
        has_end_sites=np.asarray([False, False, True]),
    )
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count, endpoint=False)
    rotations = np.zeros((frame_count, 3, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    rotations[:, 0, :] = euler_order_to_quaternion(
        "ZXY", np.stack((0.1 * np.sin(phase), 0.04 * np.cos(phase), 0.2 * np.sin(phase)), axis=1)
    )
    rotations[:, 1, :] = euler_order_to_quaternion(
        "ZXY", np.stack((0.3 * np.sin(phase), 0.2 * np.cos(phase), np.zeros_like(phase)), axis=1)
    )
    rotations[:, 2, :] = euler_order_to_quaternion(
        "ZXY", np.stack((0.5 * np.sin(phase), np.zeros_like(phase), 0.1 * np.cos(phase)), axis=1)
    )
    translations = np.broadcast_to(skeleton.offsets, (frame_count, 3, 3)).copy()
    translations[:, 0, 0] = np.linspace(0.0, 100.0, frame_count)
    translations[:, 0, 1] = 2.0 * np.sin(phase)
    return MotionClip(skeleton, rotations, translations, 1.0 / 30.0)


class CodecFormatTests(unittest.TestCase):
    def test_vectorized_rotation_quantization_matches_packed_bitstream(self) -> None:
        rng = np.random.default_rng(20260807)
        values = rng.normal(size=(257, 4))
        for bits in (8, 10, 12, 14, 16, 18, 20):
            expected = _decode_rotations(
                _encode_rotations(values, bits), len(values), bits
            )
            np.testing.assert_array_equal(
                quantize_rotation_samples(values, bits), expected
            )

    def test_full_track_quantization_can_be_reused_for_key_subsets(self) -> None:
        rng = np.random.default_rng(20260808)
        rotations = rng.normal(size=(257, 4))
        translations = rng.normal(size=257)
        indices = np.sort(rng.choice(len(rotations), size=83, replace=False))
        lower = float(np.min(translations))
        upper = float(np.max(translations))
        for bits in (8, 10, 12, 14, 16, 18, 20):
            np.testing.assert_array_equal(
                quantize_rotation_samples(rotations, bits)[indices],
                quantize_rotation_samples(rotations[indices], bits),
            )
            np.testing.assert_array_equal(
                quantize_translation_samples(translations, bits, lower, upper)[indices],
                quantize_translation_samples(
                    translations[indices], bits, lower, upper
                ),
            )

    def test_multi_threshold_rotation_keys_match_independent_runs(self) -> None:
        clip = sample_clip(37)
        track = clip.local_rotations[:, 2]
        thresholds = (0.5, 0.2, 0.05)
        expected = tuple(_rotation_key_indices(track, item) for item in thresholds)
        actual = _rotation_key_indices_multi(track, thresholds)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_unit_slerp_matches_public_normalizing_path(self) -> None:
        rng = np.random.default_rng(20260809)
        starts = rng.normal(size=(113, 4))
        ends = rng.normal(size=(113, 4))
        amounts = np.linspace(0.0, 1.0, len(starts))
        expected = quaternion_slerp(starts, ends, amounts)
        actual = _quaternion_slerp_unit(
            normalize_quaternions(starts),
            normalize_quaternions(ends),
            amounts,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_high_quality_round_trip_is_small_and_deterministic(self) -> None:
        source = sample_clip()
        settings = CodecSettings.preset("high")
        first = encode_bytes(source, settings)
        second = encode_bytes(source, settings)
        self.assertEqual(first, second)
        self.assertLess(len(first), source.raw_channel_bytes_float32)

        decoded = decode_bytes(first)
        metrics = roundtrip_metrics(source, decoded)
        self.assertLessEqual(metrics["rotation_error_degrees_max"], 0.055)
        self.assertLess(metrics["translation_error_max"], 0.001)
        self.assertLess(metrics["global_position_error_max"], 0.02)
        self.assertEqual(decoded.skeleton.signature(), source.skeleton.signature())

    def test_corruption_is_rejected(self) -> None:
        encoded = bytearray(encode_bytes(sample_clip(4)))
        encoded[-1] ^= 0x55
        with self.assertRaisesRegex(SkacFormatError, "CRC"):
            decode_bytes(bytes(encoded))

    def test_presets_trade_size_for_accuracy(self) -> None:
        source = sample_clip()
        results = {}
        for name in ("low", "medium", "high"):
            encoded = encode_bytes(source, CodecSettings.preset(name))
            results[name] = (
                len(encoded),
                roundtrip_metrics(source, decode_bytes(encoded))[
                    "rotation_error_degrees_max"
                ],
            )
        self.assertLess(results["low"][0], results["medium"][0])
        self.assertLess(results["medium"][0], results["high"][0])
        self.assertGreater(results["low"][1], results["medium"][1])
        self.assertGreater(results["medium"][1], results["high"][1])

    def test_trailing_bytes_are_rejected(self) -> None:
        encoded = encode_bytes(sample_clip(4)) + b"trailing"
        with self.assertRaisesRegex(SkacFormatError, "expected"):
            decode_bytes(encoded)


if __name__ == "__main__":
    unittest.main()
