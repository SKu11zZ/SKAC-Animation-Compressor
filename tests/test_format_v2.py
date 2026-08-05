from __future__ import annotations

import struct
import unittest
import zlib

from skac_codec.format import PREFIX, CodecSettings, SkacFormatError, decode_bytes, inspect_bytes
from skac_codec.format_v2 import encode_v2_bytes
from skac_codec.metrics import roundtrip_metrics

from tests.test_codec_format import sample_clip


class FormatV2Tests(unittest.TestCase):
    def test_segmented_round_trip_is_deterministic_and_quality_gated(self) -> None:
        source = sample_clip(frame_count=80)
        settings = CodecSettings.preset("high")
        first = encode_v2_bytes(
            source, settings, min_segment_frames=8, max_segment_frames=24
        )
        second = encode_v2_bytes(
            source, settings, min_segment_frames=8, max_segment_frames=24
        )
        self.assertEqual(first, second)
        self.assertEqual(struct.unpack_from("<H", first, 8)[0], 2)

        decoded = decode_bytes(first)
        metrics = roundtrip_metrics(source, decoded)
        self.assertLessEqual(metrics["rotation_error_degrees_max"], 0.0625)
        self.assertLessEqual(metrics["global_position_error_max"], 0.02)
        self.assertEqual(decoded.skeleton.signature(), source.skeleton.signature())

        inspected = inspect_bytes(first)
        self.assertEqual(inspected["format_major"], 2)
        self.assertGreater(inspected["codec"]["segment_count"], 1)
        self.assertEqual(
            inspected["codec"]["chunk_count"],
            1 + 2 * inspected["codec"]["segment_count"],
        )
        self.assertGreater(inspected["codec"]["translation_keyframes"], 0)

    def test_chunk_crc_rejects_damage_even_when_container_crc_is_recomputed(self) -> None:
        encoded = bytearray(
            encode_v2_bytes(sample_clip(frame_count=40), max_segment_frames=16)
        )
        fields = list(PREFIX.unpack_from(encoded))
        payload_start = PREFIX.size + int(fields[4])
        encoded[payload_start + int(fields[5]) // 2] ^= 0x40
        fields[8] = zlib.crc32(encoded[payload_start:])
        encoded[: PREFIX.size] = PREFIX.pack(*fields)
        with self.assertRaisesRegex(SkacFormatError, "chunk CRC"):
            decode_bytes(bytes(encoded))

    def test_trailing_bytes_and_invalid_segment_options_are_rejected(self) -> None:
        source = sample_clip(frame_count=12)
        encoded = encode_v2_bytes(source)
        with self.assertRaises(SkacFormatError):
            decode_bytes(encoded + b"trailing")
        with self.assertRaises(ValueError):
            encode_v2_bytes(source, min_segment_frames=1)


if __name__ == "__main__":
    unittest.main()
