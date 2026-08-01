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
from .retarget import (
    RetargetProfile,
    build_retarget_profile,
    load_retarget_profile,
    retarget_motion,
    save_retarget_profile,
)

__all__ = [
    "CodecSettings",
    "MotionClip",
    "RetargetProfile",
    "Skeleton",
    "SkacFormatError",
    "decode_bytes",
    "encode_bytes",
    "build_retarget_profile",
    "inspect_file",
    "read_bvh",
    "read_skac",
    "load_retarget_profile",
    "retarget_motion",
    "save_retarget_profile",
    "write_bvh",
    "write_skac",
]
