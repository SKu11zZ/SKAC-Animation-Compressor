from __future__ import annotations

import hashlib
import json
import math
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .format import (
    CodecSettings,
    _interpolate_rotation_track,
    _interpolate_scalar_track,
    _rotation_joint_indices,
    _rotation_key_indices_multi,
    _scalar_key_indices,
    encode_bytes,
    quantize_rotation_samples,
    quantize_translation_samples,
)
from .math3d import quaternion_angular_error_degrees
from .metrics import roundtrip_metrics
from .model import MotionClip, Skeleton
from .quality import QualityThresholds
from .retarget import _skeleton_height


PLAN_SCHEMA = "skac.adaptive_plan"
PLAN_VERSION = "1.2.0"
DEFAULT_ROTATION_BITS = (8, 10, 12, 14, 16, 18, 20)
DEFAULT_TRANSLATION_BITS = (8, 10, 12, 14, 16, 18, 20, 22, 24)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _rest_positions(skeleton: Skeleton) -> np.ndarray:
    positions = np.empty((skeleton.joint_count, 3), dtype=np.float64)
    for joint, parent in enumerate(skeleton.parents):
        positions[joint] = skeleton.offsets[joint]
        if parent >= 0:
            positions[joint] += positions[int(parent)]
    return positions


def _skeleton_scale(skeleton: Skeleton) -> float:
    positions = _rest_positions(skeleton)
    points = [positions]
    end_points = [
        positions[joint] + skeleton.end_site_offsets[joint]
        for joint, has_end in enumerate(skeleton.has_end_sites)
        if has_end
    ]
    if end_points:
        points.append(np.asarray(end_points, dtype=np.float64))
    all_points = np.concatenate(points, axis=0)
    extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    scale = float(np.linalg.norm(extent))
    if scale <= 1e-8:
        scale = float(np.max(np.linalg.norm(all_points - all_points[0], axis=1)))
    if scale <= 1e-8:
        raise ValueError("skeleton has no usable scale")
    return scale


def joint_perceptual_importance(skeleton: Skeleton) -> np.ndarray:
    """Estimate world-space error propagation from every joint to its descendants."""
    positions = _rest_positions(skeleton)
    endpoint_owner: list[int] = list(range(skeleton.joint_count))
    endpoints = [positions[joint] for joint in range(skeleton.joint_count)]
    for joint, has_end in enumerate(skeleton.has_end_sites):
        if has_end:
            endpoint_owner.append(joint)
            endpoints.append(positions[joint] + skeleton.end_site_offsets[joint])
    endpoint_array = np.asarray(endpoints, dtype=np.float64)

    ancestors: list[set[int]] = []
    for owner in endpoint_owner:
        chain: set[int] = set()
        current = owner
        while current >= 0:
            chain.add(current)
            current = int(skeleton.parents[current])
        ancestors.append(chain)

    leverage = np.zeros(skeleton.joint_count, dtype=np.float64)
    for joint in range(skeleton.joint_count):
        affected = [index for index, chain in enumerate(ancestors) if joint in chain]
        if affected:
            leverage[joint] = float(
                np.max(
                    np.linalg.norm(
                        endpoint_array[affected] - positions[joint], axis=1
                    )
                )
            )
    maximum = float(np.max(leverage))
    if maximum <= 1e-12:
        importance = np.full(skeleton.joint_count, 0.2, dtype=np.float64)
    else:
        importance = 0.2 + 0.8 * np.sqrt(np.clip(leverage / maximum, 0.0, 1.0))
    importance[0] = 1.0
    return importance


def _motion_activity(clip: MotionClip) -> np.ndarray:
    activity = np.zeros(clip.frame_count, dtype=np.float64)
    if clip.frame_count == 1:
        return activity
    rotation_delta = quaternion_angular_error_degrees(
        clip.local_rotations[1:], clip.local_rotations[:-1]
    )
    mean_rotation_speed = np.mean(rotation_delta, axis=1) / clip.frame_time
    scale = _skeleton_scale(clip.skeleton)
    root_speed = (
        np.linalg.norm(
            clip.local_translations[1:, 0] - clip.local_translations[:-1, 0], axis=1
        )
        / (scale * clip.frame_time)
    )
    activity[1:] = mean_rotation_speed + 90.0 * root_speed
    activity[0] = activity[1]
    return activity


def adaptive_segments(
    clip: MotionClip, *, min_frames: int = 8, max_frames: int = 32
) -> list[tuple[int, int]]:
    if min_frames < 2:
        raise ValueError("min_frames must be at least 2")
    if max_frames < min_frames:
        raise ValueError("max_frames must be greater than or equal to min_frames")
    if clip.frame_count <= max_frames:
        return [(0, clip.frame_count)]

    activity = _motion_activity(clip)
    median = float(np.median(activity[1:]))
    mad = float(np.median(np.abs(activity[1:] - median)))
    spike_threshold = median + max(3.0 * mad, median * 0.5, 1e-9)

    boundaries = [0]
    start = 0
    for frame in range(1, clip.frame_count):
        length = frame - start
        forced = length >= max_frames
        local_peak = (
            length >= min_frames
            and activity[frame] >= spike_threshold
            and activity[frame] > activity[frame - 1]
            and (
                frame + 1 >= clip.frame_count
                or activity[frame] >= activity[frame + 1]
            )
        )
        if forced or local_peak:
            boundaries.append(frame)
            start = frame
    if clip.frame_count - boundaries[-1] < min_frames and len(boundaries) > 1:
        previous = boundaries[-2]
        if clip.frame_count - previous <= max_frames:
            boundaries.pop()
        else:
            boundaries[-1] = clip.frame_count - min_frames
    boundaries.append(clip.frame_count)
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _varuint_size(value: int) -> int:
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


def _index_bytes(indices: np.ndarray) -> int:
    previous = 0
    size = 0
    for position, index in enumerate(np.asarray(indices, dtype=np.int64)):
        delta = int(index) if position == 0 else int(index) - previous
        size += _varuint_size(delta)
        previous = int(index)
    return size


def _plan_rotation_track(
    track: np.ndarray,
    error_budget_degrees: float,
    candidate_bits: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    options: list[tuple[int, float, int, int, float, np.ndarray, np.ndarray]] = []
    nonzero_scales = (0.65, 0.45, 0.25)
    nonzero_thresholds = tuple(
        error_budget_degrees * scale for scale in nonzero_scales
    )
    nonzero_indices = _rotation_key_indices_multi(track, nonzero_thresholds)
    index_options = (*nonzero_indices, np.arange(track.shape[0], dtype=np.int64))
    for threshold_scale, indices in zip(
        (*nonzero_scales, 0.0), index_options, strict=True
    ):
        threshold = error_budget_degrees * threshold_scale
        for bits in candidate_bits:
            values = quantize_rotation_samples(track[indices], int(bits))
            reconstructed = _interpolate_rotation_track(
                track.shape[0], indices, values
            )
            error = float(
                np.max(quaternion_angular_error_degrees(track, reconstructed))
            )
            if error <= error_budget_degrees + 1e-12:
                estimated_bits = (
                    32 + _index_bytes(indices) * 8 + len(indices) * (2 + 3 * int(bits))
                )
                options.append(
                    (
                        estimated_bits,
                        error,
                        int(bits),
                        len(indices),
                        threshold,
                        indices,
                        reconstructed,
                    )
                )
                break
    if not options:
        raise ValueError("no rotation plan satisfies the requested error budget")
    selected = min(options, key=lambda item: (item[0], item[1], item[2], item[3]))
    estimated_bits, error, bits, key_count, threshold, indices, reconstructed = selected
    record = {
        "bits": bits,
        "error_budget_degrees": float(error_budget_degrees),
        "estimated_payload_bytes_pre_entropy": (estimated_bits + 7) // 8,
        "key_count": key_count,
        "key_reduction_threshold_degrees": float(threshold),
        "reconstructed_error_degrees_max": error,
        "retained_frames_relative_to_segment": indices.tolist(),
    }
    return reconstructed, record


def _plan_translation_track(
    track: np.ndarray,
    error_budget: float,
    candidate_bits: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    lower = float(np.min(track))
    upper = float(np.max(track))
    options: list[tuple[int, float, int, int, float, np.ndarray, np.ndarray]] = []
    for threshold_scale in (0.65, 0.45, 0.25, 0.0):
        threshold = error_budget * threshold_scale
        indices = (
            np.arange(track.shape[0], dtype=np.int64)
            if threshold_scale == 0.0
            else _scalar_key_indices(track, threshold)
        )
        for bits in candidate_bits:
            values = quantize_translation_samples(
                track[indices], int(bits), lower, upper
            )
            reconstructed = _interpolate_scalar_track(
                track.shape[0], indices, values
            )
            error = float(np.max(np.abs(track - reconstructed)))
            if error <= error_budget + 1e-12:
                estimated_bits = (
                    32
                    + _index_bytes(indices) * 8
                    + len(indices) * int(bits)
                    + 128
                )
                options.append(
                    (
                        estimated_bits,
                        error,
                        int(bits),
                        len(indices),
                        threshold,
                        indices,
                        reconstructed,
                    )
                )
                break
    if not options:
        raise ValueError("no translation plan satisfies the requested error budget")
    selected = min(options, key=lambda item: (item[0], item[1], item[2], item[3]))
    estimated_bits, error, bits, key_count, threshold, indices, reconstructed = selected
    record = {
        "bits": bits,
        "error_budget": float(error_budget),
        "estimated_payload_bytes_pre_entropy": (estimated_bits + 7) // 8,
        "key_count": key_count,
        "key_reduction_threshold": float(threshold),
        "minimum": lower,
        "maximum": upper,
        "reconstructed_error_max": error,
        "retained_frames_relative_to_segment": indices.tolist(),
    }
    return reconstructed, record


def build_adaptive_plan(
    clip: MotionClip,
    *,
    settings: CodecSettings | None = None,
    min_segment_frames: int = 8,
    max_segment_frames: int = 32,
    candidate_rotation_bits: Sequence[int] = DEFAULT_ROTATION_BITS,
    candidate_translation_bits: Sequence[int] = DEFAULT_TRANSLATION_BITS,
) -> tuple[dict[str, Any], MotionClip]:
    settings = settings or CodecSettings.preset("high")
    candidate_bits = tuple(
        sorted(
            {int(item) for item in candidate_rotation_bits if int(item) <= settings.rotation_bits}
            | {settings.rotation_bits}
        )
    )
    if not candidate_bits or any(item < 8 or item > 20 for item in candidate_bits):
        raise ValueError("candidate rotation bits must be between 8 and 20")
    translation_bits = tuple(
        sorted(
            {
                int(item)
                for item in candidate_translation_bits
                if int(item) <= settings.translation_bits
            }
            | {settings.translation_bits}
        )
    )
    if not translation_bits or any(item < 8 or item > 24 for item in translation_bits):
        raise ValueError("candidate translation bits must be between 8 and 24")

    importance = joint_perceptual_importance(clip.skeleton)
    segments = adaptive_segments(
        clip, min_frames=min_segment_frames, max_frames=max_segment_frames
    )
    activity = _motion_activity(clip)
    rotation_joints = _rotation_joint_indices(clip.skeleton)
    translation_components = clip.skeleton.animated_translation_components
    thresholds = QualityThresholds.for_settings(settings)
    quality_height = _skeleton_height(clip.skeleton, 1)

    reconstructed_translations = clip.local_translations.copy()
    translation_records: list[dict[str, Any]] = []
    base_translation_budget = (
        quality_height * thresholds.max_codec_global_position_fraction * 0.1
        / math.sqrt(3.0)
    )
    for segment_index, (start, end) in enumerate(segments):
        for joint, axis in translation_components:
            weight = float(importance[joint])
            error_budget = base_translation_budget * (1.0 - 0.5 * weight)
            reconstructed, record = _plan_translation_track(
                clip.local_translations[start:end, joint, axis],
                error_budget,
                translation_bits,
            )
            reconstructed_translations[start:end, joint, axis] = reconstructed
            translation_records.append(
                {
                    "segment": segment_index,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "joint": joint,
                    "joint_name": clip.skeleton.names[joint],
                    "axis": axis,
                    "perceptual_importance": weight,
                    **record,
                }
            )

    rotation_budget_scale = 1.0
    rotation_budget_attempts = 0
    reconstructed_clip: MotionClip | None = None
    metrics: dict[str, Any] | None = None
    global_fraction = math.inf
    track_records: list[dict[str, Any]] = []
    while rotation_budget_attempts < 4:
        rotation_budget_attempts += 1
        reconstructed_rotations = clip.local_rotations.copy()
        track_records = []
        for segment_index, (start, end) in enumerate(segments):
            for joint in rotation_joints:
                weight = float(importance[joint])
                error_budget = (
                    thresholds.max_codec_rotation_degrees
                    * (1.0 - 0.4 * weight)
                    * rotation_budget_scale
                )
                reconstructed, record = _plan_rotation_track(
                    clip.local_rotations[start:end, joint],
                    error_budget,
                    candidate_bits,
                )
                reconstructed_rotations[start:end, joint] = reconstructed
                track_records.append(
                    {
                        "segment": segment_index,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "joint": joint,
                        "joint_name": clip.skeleton.names[joint],
                        "perceptual_importance": weight,
                        **record,
                    }
                )

        reconstructed_clip = MotionClip(
            skeleton=clip.skeleton,
            local_rotations=reconstructed_rotations,
            local_translations=reconstructed_translations,
            frame_time=clip.frame_time,
        )
        metrics = roundtrip_metrics(clip, reconstructed_clip)
        global_fraction = metrics["global_position_error_max"] / quality_height
        if global_fraction <= thresholds.max_codec_global_position_fraction:
            break
        ratio = thresholds.max_codec_global_position_fraction / global_fraction
        rotation_budget_scale *= max(0.35, min(0.85, ratio * 0.85))

    if reconstructed_clip is None or metrics is None:
        raise AssertionError("adaptive rotation planning did not run")
    checks = [
        {
            "id": "rotation_error_max",
            "actual": metrics["rotation_error_degrees_max"],
            "comparator": "<=",
            "limit": thresholds.max_codec_rotation_degrees,
            "unit": "degrees",
            "passed": metrics["rotation_error_degrees_max"]
            <= thresholds.max_codec_rotation_degrees,
        },
        {
            "id": "global_position_error_max",
            "actual": global_fraction,
            "comparator": "<=",
            "limit": thresholds.max_codec_global_position_fraction,
            "unit": "skeleton_scale_fraction",
            "passed": global_fraction
            <= thresholds.max_codec_global_position_fraction,
        },
    ]
    total_keys = sum(int(item["key_count"]) for item in track_records)
    weighted_bits = (
        sum(int(item["bits"]) * int(item["key_count"]) for item in track_records)
        / total_keys
        if total_keys
        else 0.0
    )
    total_translation_keys = sum(
        int(item["key_count"]) for item in translation_records
    )
    weighted_translation_bits = (
        sum(
            int(item["bits"]) * int(item["key_count"])
            for item in translation_records
        )
        / total_translation_keys
        if total_translation_keys
        else 0.0
    )
    segment_records = [
        {
            "index": index,
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
            "activity_score_mean": float(np.mean(activity[start:end])),
            "activity_score_peak": float(np.max(activity[start:end])),
        }
        for index, (start, end) in enumerate(segments)
    ]
    report: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "passed": all(bool(item["passed"]) for item in checks),
        "frame_count": clip.frame_count,
        "frame_time": clip.frame_time,
        "joint_count": clip.skeleton.joint_count,
        "skeleton_sha256": clip.skeleton.signature(),
        "quality": settings.quality_name,
        "candidate_rotation_bits": list(candidate_bits),
        "candidate_translation_bits": list(translation_bits),
        "segmentation": {
            "min_frames": min_segment_frames,
            "max_frames": max_segment_frames,
            "activity_definition": (
                "mean_joint_degrees_per_second_plus_90x_normalized_root_speed"
            ),
            "segments": segment_records,
        },
        "importance_definition": (
            "0.2_plus_0.8_sqrt_normalized_max_descendant_lever_arm;root_is_1"
        ),
        "joint_importance": [
            {
                "joint": joint,
                "joint_name": clip.skeleton.names[joint],
                "value": float(importance[joint]),
            }
            for joint in rotation_joints
        ],
        "rotation_tracks": track_records,
        "translation_tracks": translation_records,
        "summary": {
            "baseline_skac_v1_bytes": len(encode_bytes(clip, settings)),
            "planned_rotation_payload_bytes_pre_entropy": sum(
                int(item["estimated_payload_bytes_pre_entropy"])
                for item in track_records
            ),
            "planned_rotation_key_count": total_keys,
            "planned_rotation_bits_weighted_mean": float(weighted_bits),
            "rotation_track_segment_count": len(track_records),
            "planned_translation_payload_bytes_pre_entropy": sum(
                int(item["estimated_payload_bytes_pre_entropy"])
                for item in translation_records
            ),
            "planned_translation_key_count": total_translation_keys,
            "planned_translation_bits_weighted_mean": float(
                weighted_translation_bits
            ),
            "translation_track_segment_count": len(translation_records),
            "segment_count": len(segments),
            "rotation_budget_attempts": rotation_budget_attempts,
            "rotation_budget_scale": float(rotation_budget_scale),
        },
        "metrics": {
            **metrics,
            "global_position_error_max_skeleton_height_fraction": float(
                global_fraction
            ),
        },
        "checks": checks,
    }
    report["plan_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report, reconstructed_clip


def write_adaptive_plan_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_adaptive_plan_svg(path: Path, report: dict[str, Any]) -> None:
    width = 1280
    rows = sorted(
        report["joint_importance"], key=lambda item: (-item["value"], item["joint"])
    )[:20]
    height = 470 + len(rows) * 29
    summary = report["summary"]
    metrics = report["metrics"]
    status = "PASS / 通过" if report["passed"] else "REVIEW / 需检查"
    status_color = "#ff5c35" if report["passed"] else "#d13f32"

    average_bits: dict[int, float] = {}
    for row in rows:
        joint = int(row["joint"])
        tracks = [item for item in report["rotation_tracks"] if item["joint"] == joint]
        keys = sum(int(item["key_count"]) for item in tracks)
        average_bits[joint] = (
            sum(int(item["bits"]) * int(item["key_count"]) for item in tracks) / keys
            if keys
            else 0.0
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#141414}",
        ".title{font-size:32px;font-weight:700}.sub{font-size:14px;fill:#686868}",
        ".value{font-size:25px;font-weight:700}.label{font-size:12px;fill:#666}",
        ".row{font-size:12px}.small{font-size:11px;fill:#737373}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f5f4f0"/>',
        '<text x="52" y="58" class="title">SKAC v2 Perceptual Plan / 感知规划</text>',
        f'<text x="52" y="86" class="sub">Deterministic planning only · no model required · {escape(str(report["quality"]))} quality</text>',
        f'<rect x="1084" y="36" width="144" height="42" rx="21" fill="{status_color}"/>',
        f'<text x="1156" y="62" text-anchor="middle" font-size="13" font-weight="700" style="fill:#fff">{status}</text>',
    ]
    cards = [
        ("Segments / 分段", str(summary["segment_count"])),
        (
            "Mean bits R/T / 旋转·位移平均位宽",
            f'{summary["planned_rotation_bits_weighted_mean"]:.2f} / '
            f'{summary["planned_translation_bits_weighted_mean"]:.2f}',
        ),
        (
            "Keys R/T / 旋转·位移关键帧",
            f'{summary["planned_rotation_key_count"]} / '
            f'{summary["planned_translation_key_count"]}',
        ),
        ("Max rotation error / 最大旋转误差", f'{metrics["rotation_error_degrees_max"]:.5f}°'),
    ]
    for index, (label, value) in enumerate(cards):
        x = 52 + index * 294
        parts.extend(
            [
                f'<rect x="{x}" y="116" width="270" height="102" rx="12" fill="#fff" stroke="#d8d5cf"/>',
                f'<text x="{x + 20}" y="160" class="value">{escape(value)}</text>',
                f'<text x="{x + 20}" y="190" class="label">{escape(label)}</text>',
            ]
        )

    parts.extend(
        [
            '<text x="52" y="268" font-size="19" font-weight="700">Joint allocation / 关节分配</text>',
            '<text x="52" y="292" class="sub">Perceptual importance controls the error budget; bars show average selected bits.</text>',
            '<text x="52" y="313" class="small">感知权重决定误差预算，右侧色条显示各关节平均选择位宽。</text>',
        ]
    )
    chart_x = 340
    chart_width = 550
    for index, row in enumerate(rows):
        y = 350 + index * 29
        joint = int(row["joint"])
        importance = float(row["value"])
        bits = average_bits[joint]
        name = escape(str(row["joint_name"]))
        importance_width = 180 * importance
        bit_width = chart_width * bits / 20.0
        parts.extend(
            [
                f'<text x="52" y="{y + 14}" class="row">{name}</text>',
                f'<rect x="{chart_x}" y="{y}" width="180" height="16" rx="8" fill="#dfdcd5"/>',
                f'<rect x="{chart_x}" y="{y}" width="{importance_width:.2f}" height="16" rx="8" fill="#222"/>',
                f'<text x="{chart_x + 190}" y="{y + 13}" class="small">{importance:.2f}</text>',
                f'<rect x="{chart_x + 260}" y="{y}" width="{chart_width}" height="16" rx="8" fill="#e6e3dd"/>',
                f'<rect x="{chart_x + 260}" y="{y}" width="{bit_width:.2f}" height="16" rx="8" fill="#ff5c35"/>',
                f'<text x="{chart_x + 270 + bit_width:.2f}" y="{y + 13}" class="small">{bits:.1f} bit</text>',
            ]
        )
    footer_y = height - 40
    parts.extend(
        [
            f'<line x1="52" y1="{footer_y - 18}" x2="1228" y2="{footer_y - 18}" stroke="#d4d1cb"/>',
            f'<text x="52" y="{footer_y}" class="small">Planned payload covers pre-entropy rotation and translation data; it is not a final compression ratio.</text>',
            f'<text x="1228" y="{footer_y}" text-anchor="end" class="small">Plan {escape(str(report["plan_sha256"])[:12])}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
