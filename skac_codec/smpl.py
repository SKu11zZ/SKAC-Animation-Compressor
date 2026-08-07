from __future__ import annotations

from pathlib import Path

import numpy as np

from .math3d import rotation_vector_to_quaternion
from .model import MotionClip, Skeleton


# The pose stream contains the root plus 54 articulated joints. These neutral
# names describe the public SMPL-X parameter convention; no body-model files are
# embedded or required by this adapter.
SMPLX55_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye", "right_eye",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
)

SMPLX55_PARENTS = np.asarray(
    [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
        16, 17, 18, 19, 15, 15, 15,
        20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
        21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
    ],
    dtype=np.int32,
)


def _canonical_proxy_offsets() -> np.ndarray:
    """Return a normalized rest layout used only for hierarchy-aware metrics."""
    offsets = np.zeros((55, 3), dtype=np.float64)
    values = {
        1: (0.09, -0.09, 0.0), 2: (-0.09, -0.09, 0.0), 3: (0.0, 0.10, 0.0),
        4: (0.0, -0.42, 0.0), 5: (0.0, -0.42, 0.0), 6: (0.0, 0.11, 0.0),
        7: (0.0, -0.42, 0.0), 8: (0.0, -0.42, 0.0), 9: (0.0, 0.12, 0.0),
        10: (0.0, -0.04, 0.12), 11: (0.0, -0.04, 0.12), 12: (0.0, 0.16, 0.0),
        13: (0.08, 0.08, 0.0), 14: (-0.08, 0.08, 0.0), 15: (0.0, 0.14, 0.0),
        16: (0.12, 0.02, 0.0), 17: (-0.12, 0.02, 0.0),
        18: (0.25, 0.0, 0.0), 19: (-0.25, 0.0, 0.0),
        20: (0.24, 0.0, 0.0), 21: (-0.24, 0.0, 0.0),
        22: (0.0, -0.04, 0.07), 23: (0.03, 0.05, 0.08), 24: (-0.03, 0.05, 0.08),
    }
    for index, value in values.items():
        offsets[index] = value

    # Finger bases fan out from each wrist; remaining phalanges extend along X.
    base_y = (0.030, 0.010, -0.030, -0.010, -0.050)
    for side_start, direction in ((25, 1.0), (40, -1.0)):
        for finger, y in enumerate(base_y):
            base = side_start + finger * 3
            offsets[base] = (direction * 0.055, y, 0.015 if finger == 4 else 0.035)
            offsets[base + 1] = (direction * 0.035, 0.0, 0.0)
            offsets[base + 2] = (direction * 0.025, 0.0, 0.0)
    return offsets


def smplx55_proxy_skeleton() -> Skeleton:
    """Build a model-free skeleton for storing a 55-joint parameter stream.

    The offsets are a normalized proxy, not subject-specific joints. Rotation
    and root-translation Codec errors remain exact; global-position errors from
    this skeleton must be labelled as canonical-topology proxy metrics.
    """
    rotation_channels = ("Xrotation", "Yrotation", "Zrotation")
    channels = (
        ("Xposition", "Yposition", "Zposition", *rotation_channels),
        *(rotation_channels for _ in range(54)),
    )
    return Skeleton(
        names=SMPLX55_NAMES,
        parents=SMPLX55_PARENTS,
        offsets=_canonical_proxy_offsets(),
        channels=channels,
        end_site_offsets=np.zeros((55, 3), dtype=np.float64),
        has_end_sites=np.zeros(55, dtype=np.bool_),
    )


def read_smpl_npz(path: Path) -> MotionClip:
    """Load an AMASS-style 165-value SMPL-X pose stream without pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"poses", "trans", "mocap_frame_rate"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError("SMPL motion input is missing numeric animation arrays")
        poses = np.asarray(archive["poses"], dtype=np.float64)
        translations = np.asarray(archive["trans"], dtype=np.float64)
        rate_values = np.asarray(archive["mocap_frame_rate"], dtype=np.float64)

    if poses.ndim != 2 or poses.shape[1] != 165:
        raise ValueError("SMPL-X poses must have shape (frames, 165)")
    if translations.shape != (poses.shape[0], 3):
        raise ValueError("SMPL-X trans must have shape (frames, 3)")
    if rate_values.size != 1:
        raise ValueError("SMPL-X frame rate must be a numeric scalar")
    rate = float(rate_values.reshape(-1)[0])
    if poses.shape[0] < 1 or not np.isfinite(poses).all():
        raise ValueError("SMPL-X poses must be non-empty and finite")
    if not np.isfinite(translations).all():
        raise ValueError("SMPL-X translations must be finite")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("SMPL-X frame rate must be positive and finite")

    skeleton = smplx55_proxy_skeleton()
    local_translations = np.broadcast_to(
        skeleton.offsets, (poses.shape[0], 55, 3)
    ).copy()
    local_translations[:, 0] += translations
    return MotionClip(
        skeleton=skeleton,
        local_rotations=rotation_vector_to_quaternion(poses.reshape(-1, 55, 3)),
        local_translations=local_translations,
        frame_time=1.0 / rate,
    )
