"""Public reference implementation of the SKAC animation container and codec."""

from .adaptive import (
    adaptive_segments,
    build_adaptive_plan,
    joint_perceptual_importance,
    write_adaptive_plan_json,
    write_adaptive_plan_svg,
)
from .bvh import read_bvh, write_bvh
from .format import (
    CodecSettings,
    SkacFormatError,
    decode_bytes,
    encode_bytes,
    inspect_bytes,
    inspect_file,
    read_skac,
    write_skac,
)
from .format_v2 import decode_v2_bytes, encode_v2_bytes, inspect_v2_bytes, write_skac_v2
from .pack import (
    SkacPack,
    SkacPackError,
    decode_pack,
    encode_pack,
    inspect_pack_bytes,
    inspect_pack_file,
    read_pack,
    write_pack,
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
from .runtime import runtime_skeleton_bytes, runtime_skeleton_dict, save_runtime_skeleton
from .smpl import read_smpl_npz, smplx55_proxy_skeleton

__all__ = [
    "CodecSettings",
    "FbxBackendUnavailable",
    "FbxBridgeError",
    "MotionClip",
    "RetargetProfile",
    "RetargetRuntime",
    "Skeleton",
    "SkacPack",
    "SkacPackError",
    "SkacFormatError",
    "adaptive_segments",
    "build_adaptive_plan",
    "decode_bytes",
    "decode_v2_bytes",
    "decode_pack",
    "encode_bytes",
    "encode_v2_bytes",
    "encode_pack",
    "extract_fbx_to_bvh",
    "build_retarget_profile",
    "compile_retarget_profile",
    "inspect_file",
    "inspect_bytes",
    "inspect_v2_bytes",
    "inspect_pack_bytes",
    "inspect_pack_file",
    "inject_bvh_into_fbx",
    "joint_perceptual_importance",
    "read_bvh",
    "read_skac",
    "read_pack",
    "read_smpl_npz",
    "load_retarget_profile",
    "retarget_motion",
    "run_quality_gate",
    "QualityThresholds",
    "save_retarget_profile",
    "runtime_skeleton_bytes",
    "runtime_skeleton_dict",
    "save_runtime_skeleton",
    "smplx55_proxy_skeleton",
    "validate_fbx",
    "write_adaptive_plan_json",
    "write_adaptive_plan_svg",
    "write_bvh",
    "write_skac",
    "write_skac_v2",
    "write_pack",
]
