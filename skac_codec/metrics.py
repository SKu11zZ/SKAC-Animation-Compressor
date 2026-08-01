from __future__ import annotations

from typing import Any

import numpy as np

from .math3d import quaternion_angular_error_degrees, quaternion_to_matrix
from .model import MotionClip


def global_joint_positions(clip: MotionClip) -> np.ndarray:
    local_matrices = quaternion_to_matrix(clip.local_rotations)
    global_matrices = np.empty_like(local_matrices)
    global_positions = np.empty_like(clip.local_translations)
    for joint, parent in enumerate(clip.skeleton.parents):
        if parent < 0:
            global_matrices[:, joint] = local_matrices[:, joint]
            global_positions[:, joint] = clip.local_translations[:, joint]
        else:
            parent_index = int(parent)
            global_matrices[:, joint] = (
                global_matrices[:, parent_index] @ local_matrices[:, joint]
            )
            global_positions[:, joint] = global_positions[:, parent_index] + np.einsum(
                "fij,fj->fi",
                global_matrices[:, parent_index],
                clip.local_translations[:, joint],
            )
    return global_positions


def roundtrip_metrics(source: MotionClip, decoded: MotionClip) -> dict[str, Any]:
    if source.skeleton.signature() != decoded.skeleton.signature():
        raise ValueError("round-trip clips use different skeletons")
    if source.frame_count != decoded.frame_count or source.frame_time != decoded.frame_time:
        raise ValueError("round-trip clips use different timing")

    rotation_error = quaternion_angular_error_degrees(
        source.local_rotations, decoded.local_rotations
    )
    translation_error = np.linalg.norm(
        source.local_translations - decoded.local_translations, axis=-1
    )
    global_position_error = np.linalg.norm(
        global_joint_positions(source) - global_joint_positions(decoded), axis=-1
    )
    root_translation_error = np.linalg.norm(
        source.local_translations[:, 0] - decoded.local_translations[:, 0], axis=-1
    )
    return {
        "rotation_error_degrees_mean": float(np.mean(rotation_error)),
        "rotation_error_degrees_max": float(np.max(rotation_error)),
        "translation_error_mean": float(np.mean(translation_error)),
        "translation_error_max": float(np.max(translation_error)),
        "global_position_error_mean": float(np.mean(global_position_error)),
        "global_position_error_max": float(np.max(global_position_error)),
        "root_translation_error_mean": float(np.mean(root_translation_error)),
        "root_translation_error_max": float(np.max(root_translation_error)),
    }


def compression_metrics(source: MotionClip, encoded_bytes: int) -> dict[str, Any]:
    raw_bytes = source.raw_channel_bytes_float32
    return {
        "encoded_bytes": int(encoded_bytes),
        "raw_channel_bytes_float32": raw_bytes,
        "compression_ratio_vs_float32_channels": (
            float(raw_bytes / encoded_bytes) if encoded_bytes else None
        ),
        "bits_per_joint_per_frame": float(
            encoded_bytes * 8 / (source.frame_count * source.skeleton.joint_count)
        ),
    }
