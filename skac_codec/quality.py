from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .format import CodecSettings, decode_bytes, encode_bytes
from .math3d import quaternion_angular_error_degrees
from .metrics import compression_metrics, global_joint_positions, roundtrip_metrics
from .model import MotionClip, Skeleton
from .retarget import (
    PROFILE_VERSION,
    RUNTIME_MODE,
    _skeleton_height,
    build_retarget_profile,
    compile_retarget_profile,
    retarget_motion,
    retarget_motion_reference,
)


QUALITY_REPORT_SCHEMA = "skac.quality_gate"
QUALITY_REPORT_VERSION = "2.0.0"


@dataclass(frozen=True)
class QualityThresholds:
    max_codec_rotation_degrees: float
    max_codec_global_position_fraction: float
    max_runtime_rotation_degrees: float = 1e-5
    max_runtime_global_position_fraction: float = 1e-9
    max_quaternion_norm_error: float = 1e-10
    minimum_core_coverage: float = 1.0
    maximum_retarget_frame_ms_p95: float = 4.0
    minimum_decode_realtime_factor: float = 10.0
    minimum_pipeline_realtime_factor: float = 5.0

    def __post_init__(self) -> None:
        values = self.to_dict()
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("quality thresholds must be finite")
        if any(value < 0 for value in values.values()):
            raise ValueError("quality thresholds cannot be negative")

    @classmethod
    def for_settings(cls, settings: CodecSettings) -> "QualityThresholds":
        position_limits = {"low": 0.03, "medium": 0.005, "high": 0.001}
        return cls(
            max_codec_rotation_degrees=settings.rotation_error_degrees * 1.25,
            max_codec_global_position_fraction=position_limits.get(
                settings.quality_name.casefold(), 0.001
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "max_codec_rotation_degrees": self.max_codec_rotation_degrees,
            "max_codec_global_position_fraction": self.max_codec_global_position_fraction,
            "max_runtime_rotation_degrees": self.max_runtime_rotation_degrees,
            "max_runtime_global_position_fraction": self.max_runtime_global_position_fraction,
            "max_quaternion_norm_error": self.max_quaternion_norm_error,
            "minimum_core_coverage": self.minimum_core_coverage,
            "maximum_retarget_frame_ms_p95": self.maximum_retarget_frame_ms_p95,
            "minimum_decode_realtime_factor": self.minimum_decode_realtime_factor,
            "minimum_pipeline_realtime_factor": self.minimum_pipeline_realtime_factor,
        }


def _measure_ms(call: Callable[[], object], iterations: int) -> list[float]:
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        call()
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return durations


def _percentile(values: list[float], amount: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), amount))


def _check(
    identifier: str,
    category: str,
    actual: float,
    comparator: str,
    limit: float,
    unit: str,
) -> dict[str, Any]:
    if comparator == "<=":
        passed = actual <= limit
    elif comparator == ">=":
        passed = actual >= limit
    else:
        raise ValueError("unsupported quality-gate comparator")
    return {
        "id": identifier,
        "category": category,
        "actual": float(actual),
        "comparator": comparator,
        "limit": float(limit),
        "unit": unit,
        "passed": bool(passed),
    }


def run_quality_gate(
    source: MotionClip,
    target_skeleton: Skeleton,
    *,
    settings: CodecSettings | None = None,
    thresholds: QualityThresholds | None = None,
    decode_iterations: int = 5,
    pipeline_iterations: int = 3,
    frame_samples: int = 300,
    evaluation_case: str = "auto",
) -> dict[str, Any]:
    settings = settings or CodecSettings.preset("high")
    thresholds = thresholds or QualityThresholds.for_settings(settings)
    if decode_iterations < 1 or pipeline_iterations < 1 or frame_samples < 1:
        raise ValueError("quality-gate iteration counts must be positive")

    requested_case = evaluation_case.replace("-", "_").casefold()
    if requested_case not in {"auto", "same_character", "different_character"}:
        raise ValueError(
            "evaluation_case must be auto, same_character, or different_character"
        )
    same_skeleton = source.skeleton.signature() == target_skeleton.signature()
    resolved_case = (
        "same_character" if same_skeleton else "different_character"
    ) if requested_case == "auto" else requested_case
    if resolved_case == "same_character" and not same_skeleton:
        raise ValueError("same_character requires identical source and target skeletons")

    encoded = encode_bytes(source, settings)
    decoded = decode_bytes(encoded)
    codec_quality = roundtrip_metrics(source, decoded)
    source_height = _skeleton_height(source.skeleton, 1)
    playback_budget_seconds = source.frame_count * source.frame_time
    decode_times = _measure_ms(lambda: decode_bytes(encoded), decode_iterations)
    decode_median_ms = float(statistics.median(decode_times))
    decode_realtime_factor = playback_budget_seconds / (decode_median_ms / 1000.0)
    codec_global_fraction = codec_quality["global_position_error_max"] / source_height
    decoded_quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(decoded.local_rotations, axis=-1) - 1.0))
    )
    checks = [
        _check(
            "codec_rotation",
            "quality",
            codec_quality["rotation_error_degrees_max"],
            "<=",
            thresholds.max_codec_rotation_degrees,
            "degrees",
        ),
        _check(
            "codec_global_position",
            "quality",
            codec_global_fraction,
            "<=",
            thresholds.max_codec_global_position_fraction,
            "skeleton_height_fraction",
        ),
        _check(
            "codec_quaternion_norm",
            "quality",
            decoded_quaternion_norm_error,
            "<=",
            thresholds.max_quaternion_norm_error,
            "absolute",
        ),
        _check(
            "decode_realtime_factor",
            "performance",
            decode_realtime_factor,
            ">=",
            thresholds.minimum_decode_realtime_factor,
            "x_realtime",
        ),
    ]

    profile_record: dict[str, Any] | None = None
    runtime_equivalence: dict[str, Any] | None = None
    runtime_diagnostics: dict[str, Any] | None = None
    performance: dict[str, Any] = {
        "decode_clip_ms_median": decode_median_ms,
        "decode_realtime_factor": decode_realtime_factor,
        "decode_iterations": decode_iterations,
    }
    if resolved_case == "different_character":
        profile = build_retarget_profile(
            decoded.skeleton, target_skeleton, contact_lock=False
        )
        compile_started = time.perf_counter_ns()
        runtime = compile_retarget_profile(decoded.skeleton, target_skeleton, profile)
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
        runtime_output, runtime_diagnostics = retarget_motion(
            decoded, target_skeleton, profile, runtime=runtime
        )
        reference_output, _ = retarget_motion_reference(
            decoded, target_skeleton, profile
        )
        runtime_rotation_error = quaternion_angular_error_degrees(
            runtime_output.local_rotations, reference_output.local_rotations
        )
        runtime_position_error = np.linalg.norm(
            global_joint_positions(runtime_output)
            - global_joint_positions(reference_output),
            axis=-1,
        )
        runtime_quaternion_norm_error = float(
            np.max(
                np.abs(np.linalg.norm(runtime_output.local_rotations, axis=-1) - 1.0)
            )
        )
        target_height = _skeleton_height(target_skeleton, profile.up_axis)
        runtime_global_fraction = float(np.max(runtime_position_error)) / target_height

        output_rotations = np.empty((target_skeleton.joint_count, 4), dtype=np.float64)
        output_translations = np.empty((target_skeleton.joint_count, 3), dtype=np.float64)
        for frame in range(min(source.frame_count, 16)):
            runtime.evaluate_frame_into(
                decoded.local_rotations[frame],
                decoded.local_translations[frame, 0],
                output_rotations,
                output_translations,
            )
        frame_times: list[float] = []
        frame_batch_size = min(10, frame_samples)
        measured_samples = 0
        while measured_samples < frame_samples:
            current_batch = min(frame_batch_size, frame_samples - measured_samples)
            started = time.perf_counter_ns()
            for offset in range(current_batch):
                frame = (measured_samples + offset) % source.frame_count
                runtime.evaluate_frame_into(
                    decoded.local_rotations[frame],
                    decoded.local_translations[frame, 0],
                    output_rotations,
                    output_translations,
                )
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            frame_times.append(elapsed_ms / current_batch)
            measured_samples += current_batch

        def pipeline() -> None:
            clip = decode_bytes(encoded)
            retarget_motion(clip, target_skeleton, profile, runtime=runtime)

        pipeline_times = _measure_ms(pipeline, pipeline_iterations)
        pipeline_median_ms = float(statistics.median(pipeline_times))
        retarget_frame_p95_ms = _percentile(frame_times, 95.0)
        pipeline_realtime_factor = playback_budget_seconds / (
            pipeline_median_ms / 1000.0
        )
        checks.extend(
            [
                _check(
                    "profile_core_coverage",
                    "quality",
                    profile.core_coverage,
                    ">=",
                    thresholds.minimum_core_coverage,
                    "ratio",
                ),
                _check(
                    "runtime_rotation_equivalence",
                    "quality",
                    float(np.max(runtime_rotation_error)),
                    "<=",
                    thresholds.max_runtime_rotation_degrees,
                    "degrees",
                ),
                _check(
                    "runtime_global_equivalence",
                    "quality",
                    runtime_global_fraction,
                    "<=",
                    thresholds.max_runtime_global_position_fraction,
                    "skeleton_height_fraction",
                ),
                _check(
                    "runtime_quaternion_norm",
                    "quality",
                    runtime_quaternion_norm_error,
                    "<=",
                    thresholds.max_quaternion_norm_error,
                    "absolute",
                ),
                _check(
                    "retarget_frame_p95",
                    "performance",
                    retarget_frame_p95_ms,
                    "<=",
                    thresholds.maximum_retarget_frame_ms_p95,
                    "milliseconds",
                ),
                _check(
                    "pipeline_realtime_factor",
                    "performance",
                    pipeline_realtime_factor,
                    ">=",
                    thresholds.minimum_pipeline_realtime_factor,
                    "x_realtime",
                ),
            ]
        )
        profile_record = {
            "schema_version": PROFILE_VERSION,
            "profile_sha256": profile.signature(),
            "runtime_mode": RUNTIME_MODE,
            "mapped_joint_count": len(profile.transfers),
            "shared_core_joint_count": profile.shared_core_joint_count,
            "mapped_core_joint_count": profile.mapped_core_joint_count,
            "core_coverage": profile.core_coverage,
            "source_runtime_joint_count": len(runtime.source_evaluation_order),
            "target_runtime_joint_count": len(runtime.target_evaluation_order),
            "compile_ms": compile_ms,
        }
        runtime_equivalence = {
            "rotation_error_degrees_max": float(np.max(runtime_rotation_error)),
            "global_position_error_max": float(np.max(runtime_position_error)),
            "global_position_error_max_height_fraction": runtime_global_fraction,
            "quaternion_norm_error_max": runtime_quaternion_norm_error,
        }
        performance.update(
            {
                "retarget_frame_ms_p50": _percentile(frame_times, 50.0),
                "retarget_frame_ms_p95": retarget_frame_p95_ms,
                "retarget_fps_at_p95": 1000.0 / retarget_frame_p95_ms,
                "pipeline_clip_ms_median": pipeline_median_ms,
                "pipeline_realtime_factor": pipeline_realtime_factor,
                "pipeline_iterations": pipeline_iterations,
                "frame_samples": frame_samples,
                "frame_batch_size": frame_batch_size,
            }
        )

    passed = all(item["passed"] for item in checks)
    applicable_thresholds = thresholds.to_dict()
    if resolved_case == "same_character":
        applicable_thresholds = {
            key: applicable_thresholds[key]
            for key in (
                "max_codec_rotation_degrees",
                "max_codec_global_position_fraction",
                "max_quaternion_norm_error",
                "minimum_decode_realtime_factor",
            )
        }
    return {
        "schema": QUALITY_REPORT_SCHEMA,
        "schema_version": QUALITY_REPORT_VERSION,
        "passed": passed,
        "evaluation_case": resolved_case,
        "case_classification": (
            "skeleton_signature_equality" if requested_case == "auto" else "explicit"
        ),
        "scope": {
            "pipeline": (
                "bvh_encode_decode_direct_source_skeleton_playback"
                if resolved_case == "same_character"
                else "bvh_encode_decode_then_compiled_target_playback"
            ),
            "retarget_ground_truth_available": (
                None if resolved_case == "same_character" else False
            ),
            "quality_claim": (
                "codec reconstruction and direct source-skeleton decode performance"
                if resolved_case == "same_character"
                else "codec reconstruction, compiled-runtime equivalence, and shared-core "
                "mapping coverage; not perceptual target-motion ground truth"
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": sys.platform,
        },
        "source": {
            "skeleton_sha256": source.skeleton.signature(),
            "joint_count": source.skeleton.joint_count,
            "frame_count": source.frame_count,
            "frame_time": source.frame_time,
            "duration_seconds": source.duration_seconds,
            "playback_budget_seconds": playback_budget_seconds,
        },
        "target": (
            None
            if resolved_case == "same_character"
            else {
                "skeleton_sha256": target_skeleton.signature(),
                "joint_count": target_skeleton.joint_count,
            }
        ),
        "codec": {
            "quality": settings.quality_name,
            **compression_metrics(source, len(encoded)),
            **codec_quality,
            "global_position_error_max_height_fraction": codec_global_fraction,
            "quaternion_norm_error_max": decoded_quaternion_norm_error,
        },
        "profile": profile_record,
        "runtime_equivalence": runtime_equivalence,
        "performance": performance,
        "thresholds": applicable_thresholds,
        "checks": checks,
        "runtime_diagnostics": runtime_diagnostics,
    }


def write_quality_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_quality_report_svg(path: Path, report: dict[str, Any]) -> None:
    checks = report["checks"]
    width = 1280
    row_height = 48
    height = 190 + len(checks) * row_height
    passed = bool(report["passed"])
    status_color = "#15803d" if passed else "#b91c1c"
    background = "#f8fafc"
    same_character = report["evaluation_case"] == "same_character"
    case_label = (
        "Same character - direct Codec decode"
        if same_character
        else "Different character - compiled Profile playback"
    )
    profile_label = (
        "No Profile cost on this route"
        if same_character
        else f'Profile {report["profile"]["schema_version"]} - {report["profile"]["runtime_mode"]}'
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif}.title{font-size:28px;font-weight:700}.small{font-size:13px;fill:#475569}.label{font-size:14px;font-weight:600;fill:#0f172a}.value{font-size:13px;fill:#334155}</style>',
        f'<text x="48" y="52" class="title">SKAC {escape(case_label)} Gate</text>',
        f'<rect x="1060" y="24" width="170" height="46" rx="23" fill="{status_color}"/>',
        f'<text x="1145" y="54" text-anchor="middle" font-size="20" font-weight="700" fill="white">{"PASS" if passed else "FAIL"}</text>',
        f'<text x="48" y="82" class="small">{escape(profile_label)}</text>',
        '<text x="48" y="108" class="small">Green bars pass the frozen threshold. Red bars block the build.</text>',
    ]
    y = 150
    for check in checks:
        actual = float(check["actual"])
        limit = float(check["limit"])
        if check["comparator"] == "<=":
            ratio = actual / limit if limit > 0 else (0.0 if actual <= 0 else 2.0)
        else:
            ratio = limit / actual if actual > 0 else 2.0
        fill_width = min(420.0, 420.0 * max(0.0, ratio))
        color = "#22c55e" if check["passed"] else "#ef4444"
        label = escape(str(check["id"]).replace("_", " "))
        value = escape(
            f'{actual:.6g} {check["unit"]} {check["comparator"]} {limit:.6g}'
        )
        lines.extend(
            [
                f'<text x="48" y="{y + 17}" class="label">{label}</text>',
                f'<rect x="390" y="{y}" width="420" height="22" rx="5" fill="#e2e8f0"/>',
                f'<rect x="390" y="{y}" width="{fill_width:.2f}" height="22" rx="5" fill="{color}"/>',
                f'<text x="830" y="{y + 17}" class="value">{value}</text>',
            ]
        )
        y += row_height
    lines.extend(
        [
            f'<text x="48" y="{height - 26}" class="small">Quality scope: {escape(str(report["scope"]["quality_claim"]))}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
