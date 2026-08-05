from __future__ import annotations

import unittest

from skac_codec.bvh import loads_bvh
from skac_codec.format import CodecSettings, encode_bytes
from skac_codec.pack import (
    SkacPackError,
    decode_pack,
    encode_pack,
    inspect_pack_bytes,
)


SAMPLE_BVH = """HIERARCHY
ROOT Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  End Site
  {
    OFFSET 0 1 0
  }
}
MOTION
Frames: 4
Frame Time: 0.0333333333
0 0 0 0 0 0
1 0 0 5 0 0
2 0 0 10 0 0
3 0 0 15 0 0
"""


def sample_clip():
    return loads_bvh(SAMPLE_BVH)


class SkacPackTests(unittest.TestCase):
    def test_pack_is_deterministic_and_deduplicates_identical_clips(self) -> None:
        encoded = encode_bytes(sample_clip(), CodecSettings.preset("high"))
        first = encode_pack({"walk": encoded, "walk-copy": encoded})
        second = encode_pack({"walk-copy": encoded, "walk": encoded})
        self.assertEqual(first, second)

        report = inspect_pack_bytes(first)
        self.assertEqual(report["entry_count"], 2)
        self.assertEqual(report["unique_blob_count"], 1)
        self.assertEqual(report["deduplicated_entry_count"], 1)

        archive = decode_pack(first)
        self.assertEqual(archive.entry_ids, ("walk", "walk-copy"))
        self.assertEqual(archive.entry_bytes("walk"), encoded)
        self.assertEqual(archive.entry_bytes("walk-copy"), encoded)

    def test_pack_preserves_distinct_clips(self) -> None:
        high = encode_bytes(sample_clip(), CodecSettings.preset("high"))
        low = encode_bytes(sample_clip(), CodecSettings.preset("low"))
        report = inspect_pack_bytes(encode_pack({"high": high, "low": low}))
        self.assertEqual(report["unique_blob_count"], 2)
        self.assertEqual(report["deduplicated_entry_count"], 0)

    def test_pack_rejects_corruption_and_unsafe_ids(self) -> None:
        encoded = encode_bytes(sample_clip())
        with self.assertRaisesRegex(SkacPackError, "entry id"):
            encode_pack({"../walk": encoded})

        damaged = bytearray(encode_pack({"walk": encoded}))
        damaged[-1] ^= 0x55
        with self.assertRaisesRegex(SkacPackError, "CRC"):
            decode_pack(bytes(damaged))


if __name__ == "__main__":
    unittest.main()
