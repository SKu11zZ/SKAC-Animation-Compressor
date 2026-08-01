"""Public reference implementation of the SKAC animation container and codec."""

from .bvh import read_bvh, write_bvh
from .format import (
    CodecSettings,
    SkacFormatError,
    decode_bytes,
    encode_bytes,
    inspect_file,
    read_skac,
    write_skac,
)
from .model import MotionClip, Skeleton

__all__ = [
    "CodecSettings",
    "MotionClip",
    "Skeleton",
    "SkacFormatError",
    "decode_bytes",
    "encode_bytes",
    "inspect_file",
    "read_bvh",
    "read_skac",
    "write_bvh",
    "write_skac",
]
