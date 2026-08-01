"""Public, dependency-light cross-skeleton motion retargeting core."""

from .retarget import (
    Motion,
    RetargetProfile,
    Skeleton,
    build_profile,
    canonical_joint_name,
    retarget_motion,
)

__all__ = [
    "Motion",
    "RetargetProfile",
    "Skeleton",
    "build_profile",
    "canonical_joint_name",
    "retarget_motion",
]
