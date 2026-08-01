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
from .fbx import (
    FbxBackendUnavailable,
    FbxBridgeError,
    extract_fbx_to_bvh,
    inject_bvh_into_fbx,
    validate_fbx,
)
from .model import MotionClip, Skeleton
from .quality import QualityThresholds, run_quality_gate
from .retarget import (
    RetargetProfile,
    RetargetRuntime,
    build_retarget_profile,
    compile_retarget_profile,
    load_retarget_profile,
    retarget_motion,
    save_retarget_profile,
)

__all__ = [
    "CodecSettings",
    "FbxBackendUnavailable",
    "FbxBridgeError",
    "MotionClip",
    "RetargetProfile",
    "RetargetRuntime",
    "Skeleton",
    "SkacFormatError",
    "decode_bytes",
    "encode_bytes",
    "extract_fbx_to_bvh",
    "build_retarget_profile",
    "compile_retarget_profile",
    "inspect_file",
    "inject_bvh_into_fbx",
    "read_bvh",
    "read_skac",
    "load_retarget_profile",
    "retarget_motion",
    "run_quality_gate",
    "QualityThresholds",
    "save_retarget_profile",
    "validate_fbx",
    "write_bvh",
    "write_skac",
]
