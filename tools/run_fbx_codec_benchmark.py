from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from skac_codec.bvh import read_bvh, write_bvh  # noqa: E402
from skac_codec.fbx import (  # noqa: E402
    extract_fbx_to_bvh,
    inject_bvh_into_fbx,
    validate_fbx,
)
from skac_codec.format import CodecSettings, decode_bytes, encode_bytes  # noqa: E402
from skac_codec.metrics import compression_metrics, roundtrip_metrics  # noqa: E402
from skac_codec.model import MotionClip  # noqa: E402
from skac_codec.quality import QualityThresholds  # noqa: E402
from skac_codec.retarget import _skeleton_height  # noqa: E402


SCHEMA = "skac.fbx_codec_benchmark"
SCHEMA_VERSION = "1.0.0"
FAMILIES = ("mixamo", "manny")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract neutral public FBX motions and benchmark SKAC v1."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bvh-cache", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--extract-workers", type=int, default=2)
    parser.add_argument("--codec-workers", type=int, default=4)
    parser.add_argument("--decode-iterations", type=int, default=3)
    return parser.parse_args(argv)


def _inventory(root: Path) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for family in FAMILIES:
        family_root = root / family
        if not family_root.is_dir():
            raise ValueError(f"missing neutral public cache family: {family}")
        files = sorted(family_root.glob(f"{family}_*.fbx"))
        if not files:
            raise ValueError(f"neutral public cache family has no FBX files: {family}")
        records.extend((family, path.stem, path) for path in files)
    return records


def _blender_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--background", "--factory-startup", "--version"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Blender version probe failed")
    first = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first.startswith("Blender "):
        raise RuntimeError("Blender version probe returned an unexpected response")
    return first


def _extract(job: tuple[str, str, str, str, str]) -> dict[str, Any]:
    family, identifier, source_text, output_text, blender_text = job
    output = Path(output_text)
    cache_hit = output.is_file() and output.stat().st_size > 0
    if cache_hit:
        clip = read_bvh(output)
        return {
            "family": family,
            "id": identifier,
            "cache_hit": True,
            "source_frame_count": clip.frame_count,
            "joint_count": clip.skeleton.joint_count,
        }
    report = extract_fbx_to_bvh(
        Path(source_text),
        output,
        blender_executable=Path(blender_text),
        timeout_seconds=600,
    )
    clip = read_bvh(output)
    return {
        "family": family,
        "id": identifier,
        "cache_hit": False,
        "source_frame_count": clip.frame_count,
        "joint_count": clip.skeleton.joint_count,
        "armature_count": int(report.get("armature_count", 0)),
        "bone_count": int(report.get("bone_count", 0)),
        "mesh_count": int(report.get("mesh_count", 0)),
        "material_count": int(report.get("material_count", 0)),
        "action_count": int(report.get("action_count", 0)),
        "keyframe_count": int(report.get("keyframe_count", 0)),
    }


def _measure(job: tuple[str, str, str, int, int, CodecSettings]) -> dict[str, Any]:
    family, identifier, source_text, max_frames, decode_iterations, settings = job
    full_clip = read_bvh(Path(source_text))
    start = max(0, (full_clip.frame_count - max_frames) // 2)
    end = min(full_clip.frame_count, start + max_frames)
    clip = MotionClip(
        skeleton=full_clip.skeleton,
        local_rotations=full_clip.local_rotations[start:end],
        local_translations=full_clip.local_translations[start:end],
        frame_time=full_clip.frame_time,
    )
    started = time.perf_counter_ns()
    encoded = encode_bytes(clip, settings)
    encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    decode_bytes(encoded)
    decode_durations: list[float] = []
    decoded = None
    for _ in range(decode_iterations):
        started = time.perf_counter_ns()
        decoded = decode_bytes(encoded)
        decode_durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert decoded is not None
    metrics = roundtrip_metrics(clip, decoded)
    height = _skeleton_height(clip.skeleton, 1)
    return {
        "family": family,
        "id": identifier,
        "source_frame_count": full_clip.frame_count,
        "window_start_frame": start,
        "frame_count": clip.frame_count,
        "frame_time": clip.frame_time,
        "playback_seconds": clip.frame_count * clip.frame_time,
        "joint_count": clip.skeleton.joint_count,
        "encode_clip_ms": encode_ms,
        "decode_clip_ms_median": statistics.median(decode_durations),
        **compression_metrics(clip, len(encoded)),
        **metrics,
        "global_position_error_max_height_fraction": (
            metrics["global_position_error_max"] / height
        ),
        "root_translation_error_max_height_fraction": (
            metrics["root_translation_error_max"] / height
        ),
    }


def _weighted_mean(samples: list[dict[str, Any]], field: str) -> float:
    return float(
        np.average(
            [float(item[field]) for item in samples],
            weights=[int(item["frame_count"]) * int(item["joint_count"]) for item in samples],
        )
    )


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot aggregate an empty sample")
    raw_bytes = sum(int(item["raw_channel_bytes_float32"]) for item in samples)
    encoded_bytes = sum(int(item["encoded_bytes"]) for item in samples)
    playback = sum(float(item["playback_seconds"]) for item in samples)
    encode_seconds = sum(float(item["encode_clip_ms"]) for item in samples) / 1000.0
    decode_seconds = sum(float(item["decode_clip_ms_median"]) for item in samples) / 1000.0
    return {
        "motion_count": len(samples),
        "source_frame_count": sum(int(item["source_frame_count"]) for item in samples),
        "evaluated_frame_count": sum(int(item["frame_count"]) for item in samples),
        "playback_seconds": playback,
        "raw_channel_bytes_float32": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "compression_ratio_vs_float32_channels": raw_bytes / encoded_bytes,
        "encode_seconds_sum": encode_seconds,
        "decode_seconds_median_sum": decode_seconds,
        "encode_realtime_factor": playback / encode_seconds,
        "decode_realtime_factor": playback / decode_seconds,
        "rotation_error_degrees_mean_weighted": _weighted_mean(
            samples, "rotation_error_degrees_mean"
        ),
        "rotation_error_degrees_max": max(
            float(item["rotation_error_degrees_max"]) for item in samples
        ),
        "root_translation_error_max": max(
            float(item["root_translation_error_max"]) for item in samples
        ),
        "global_position_error_max": max(
            float(item["global_position_error_max"]) for item in samples
        ),
        "global_position_error_max_height_fraction": max(
            float(item["global_position_error_max_height_fraction"]) for item in samples
        ),
        "root_translation_error_max_height_fraction": max(
            float(item["root_translation_error_max_height_fraction"]) for item in samples
        ),
        "joint_count_min": min(int(item["joint_count"]) for item in samples),
        "joint_count_max": max(int(item["joint_count"]) for item in samples),
    }


def _fmt_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):,.2f} MiB"


def _roundtrip_smoke(
    inventory: list[tuple[str, str, Path]],
    bvh_cache: Path,
    blender: Path,
    settings: CodecSettings,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="skac-fbx-roundtrip-") as temporary:
        root = Path(temporary)
        for family in FAMILIES:
            _, identifier, source_fbx = next(
                item for item in inventory if item[0] == family
            )
            try:
                clip = read_bvh(bvh_cache / family / f"{identifier}.bvh")
                decoded = decode_bytes(encode_bytes(clip, settings))
                restored_bvh = root / f"{identifier}.bvh"
                restored_fbx = root / f"{identifier}.fbx"
                write_bvh(restored_bvh, decoded)
                injected = inject_bvh_into_fbx(
                    source_fbx,
                    restored_bvh,
                    restored_fbx,
                    blender_executable=blender,
                    timeout_seconds=600,
                )
                validated = validate_fbx(
                    restored_fbx,
                    blender_executable=blender,
                    timeout_seconds=300,
                )
                samples.append(
                    {
                        "family": family,
                        "id": identifier,
                        "passed": True,
                        "frame_count": clip.frame_count,
                        "joint_count": clip.skeleton.joint_count,
                        "mapped_bone_count": int(injected.get("mapped_bone_count", 0)),
                        "output_bone_count": int(validated.get("bone_count", 0)),
                        "output_action_count": int(validated.get("action_count", 0)),
                        "output_keyframe_count": int(validated.get("keyframe_count", 0)),
                        "output_mesh_count": int(validated.get("mesh_count", 0)),
                        "output_material_count": int(validated.get("material_count", 0)),
                        "missing_external_dependency_count": len(
                            validated.get("missing_external_dependencies", [])
                        ),
                    }
                )
            except Exception as error:
                samples.append(
                    {
                        "family": family,
                        "id": identifier,
                        "passed": False,
                        "error_type": type(error).__name__,
                    }
                )
    return {
        "sample_count": len(samples),
        "passed": all(bool(item["passed"]) for item in samples),
        "selection": "first_neutral_id_per_family",
        "assets_in_report": False,
        "samples": samples,
    }


def _quality_checks(
    aggregate: dict[str, Any], settings: CodecSettings
) -> list[dict[str, Any]]:
    thresholds = QualityThresholds.for_settings(settings)
    definitions = (
        (
            "maximum_local_rotation_error",
            float(aggregate["rotation_error_degrees_max"]),
            thresholds.max_codec_rotation_degrees,
            "degrees",
            "maximum",
        ),
        (
            "maximum_global_position_error",
            float(aggregate["global_position_error_max_height_fraction"]),
            thresholds.max_codec_global_position_fraction,
            "skeleton_height_fraction",
            "maximum",
        ),
        (
            "minimum_decode_realtime_factor",
            float(aggregate["decode_realtime_factor"]),
            thresholds.minimum_decode_realtime_factor,
            "x_realtime",
            "minimum",
        ),
    )
    return [
        {
            "id": identifier,
            "actual": actual,
            "limit": limit,
            "unit": unit,
            "comparison": comparison,
            "passed": actual <= limit if comparison == "maximum" else actual >= limit,
        }
        for identifier, actual, limit, unit, comparison in definitions
    ]


def _svg(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    families = report["families"]
    cards = [
        ("FBX MOTIONS / 动画", f'{aggregate["motion_count"]}'),
        ("EVALUATED FRAMES / 实测帧", f'{aggregate["evaluated_frame_count"]:,}'),
        ("RAW CHANNELS / 原始通道", _fmt_bytes(aggregate["raw_channel_bytes_float32"])),
        ("SKAC FILES / 压缩后", _fmt_bytes(aggregate["encoded_bytes"])),
        ("COMPRESSION / 压缩比", f'{aggregate["compression_ratio_vs_float32_channels"]:.2f}×'),
        ("MAX ROTATION ERROR / 最大旋转误差", f'{aggregate["rotation_error_degrees_max"]:.4f}°'),
    ]
    card_markup: list[str] = []
    for index, (label, value) in enumerate(cards):
        col, row = index % 3, index // 3
        x, y = 54 + col * 398, 164 + row * 124
        card_markup.append(
            f'<rect x="{x}" y="{y}" width="370" height="96" rx="12" class="card"/>'
            f'<text x="{x + 20}" y="{y + 29}" class="label">{escape(label)}</text>'
            f'<text x="{x + 20}" y="{y + 69}" class="value">{escape(value)}</text>'
        )
    max_raw = max(int(item["raw_channel_bytes_float32"]) for item in families)
    bars: list[str] = []
    for index, item in enumerate(families):
        y = 478 + index * 78
        raw_width = 720.0 * int(item["raw_channel_bytes_float32"]) / max_raw
        encoded_width = 720.0 * int(item["encoded_bytes"]) / max_raw
        bars.append(
            f'<text x="54" y="{y + 22}" class="axis">{escape(str(item["family"]).title())}</text>'
            f'<rect x="190" y="{y}" width="{raw_width:.1f}" height="20" rx="4" class="raw"/>'
            f'<rect x="190" y="{y + 28}" width="{encoded_width:.1f}" height="20" rx="4" class="accent"/>'
            f'<text x="{200 + raw_width:.1f}" y="{y + 16}" class="axis">{escape(_fmt_bytes(int(item["raw_channel_bytes_float32"])))}</text>'
            f'<text x="{200 + encoded_width:.1f}" y="{y + 44}" class="axis">{escape(_fmt_bytes(int(item["encoded_bytes"])))}</text>'
        )
    gate_label = "QUALITY GATE: PASS" if report["passed"] else "QUALITY GATE: FAIL"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<style>.bg{{fill:#0e1012}}.card{{fill:#171a1e;stroke:#30343a}}.title{{fill:#f3f4f6;font:700 34px Arial,sans-serif}}.sub{{fill:#979da7;font:15px Arial,sans-serif}}.label{{fill:#9da3ad;font:700 12px Arial,sans-serif;letter-spacing:.4px}}.value{{fill:#f6f7f8;font:700 28px Arial,sans-serif}}.axis{{fill:#abb1ba;font:13px Arial,sans-serif}}.note{{fill:#858c96;font:12px Arial,sans-serif}}.gate{{fill:#ff5a36;font:700 13px Arial,sans-serif;letter-spacing:.6px}}.raw{{fill:#e9eaec}}.accent{{fill:#ff5a36}}</style>
<rect width="1280" height="720" class="bg"/><rect width="12" height="720" class="accent"/>
<text x="54" y="62" class="title">SKAC v1 · Public FBX Codec Benchmark</text>
<text x="54" y="94" class="sub">Mixamo + Manny · Blender 4.5 LTS extraction · high quality / 公开 FBX 动画压缩测试</text>
<text x="1220" y="94" text-anchor="end" class="gate">{gate_label}</text>
<text x="54" y="121" class="sub">{aggregate["source_frame_count"]:,} source frames · {aggregate["evaluated_frame_count"]:,} deterministic center-window frames · no FBX assets shipped</text>
{''.join(card_markup)}
<text x="54" y="448" class="label">STORAGE BY FAMILY / 各动画族存储量　　WHITE = FLOAT32　ORANGE = SKAC</text>
{''.join(bars)}
<text x="54" y="660" class="note">Summed encode {aggregate["encode_seconds_sum"]:.2f} s ({aggregate["encode_realtime_factor"]:.2f}× realtime) · Python decode {aggregate["decode_seconds_median_sum"]:.2f} s ({aggregate["decode_realtime_factor"]:.1f}× realtime)</text>
<text x="54" y="686" class="note">Backend: {escape(str(report["backend"]["blender_version"]))} · Python {escape(str(report["environment"]["python"]))} · anonymous neutral IDs only</text>
</svg>'''


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("max_frames", "extract_workers", "codec_workers", "decode_iterations"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    blender = args.blender.resolve()
    if not blender.is_file():
        raise ValueError("the explicit Blender executable is missing")
    inventory = _inventory(args.data_root)
    args.bvh_cache.mkdir(parents=True, exist_ok=True)
    version = _blender_version(blender)

    extract_started = time.perf_counter()
    extract_jobs = [
        (
            family,
            identifier,
            str(source),
            str(args.bvh_cache / family / f"{identifier}.bvh"),
            str(blender),
        )
        for family, identifier, source in inventory
    ]
    for family in FAMILIES:
        (args.bvh_cache / family).mkdir(parents=True, exist_ok=True)
    extraction: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.extract_workers) as executor:
        futures = {executor.submit(_extract, job): job[1] for job in extract_jobs}
        for index, future in enumerate(as_completed(futures), 1):
            extraction.append(future.result())
            print(f"extracted {index}/{len(futures)}", flush=True)
    extraction.sort(key=lambda item: (str(item["family"]), str(item["id"])))
    extraction_seconds = time.perf_counter() - extract_started

    settings = CodecSettings.preset(args.quality)
    codec_jobs = [
        (
            family,
            identifier,
            str(args.bvh_cache / family / f"{identifier}.bvh"),
            args.max_frames,
            args.decode_iterations,
            settings,
        )
        for family, identifier, _ in inventory
    ]
    codec_started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.codec_workers) as executor:
        futures = {executor.submit(_measure, job): job[1] for job in codec_jobs}
        for index, future in enumerate(as_completed(futures), 1):
            samples.append(future.result())
            print(f"encoded {index}/{len(futures)}", flush=True)
    samples.sort(key=lambda item: (str(item["family"]), str(item["id"])))
    codec_wall_seconds = time.perf_counter() - codec_started

    family_reports = [
        {"family": family, **_aggregate([item for item in samples if item["family"] == family])}
        for family in FAMILIES
    ]
    aggregate = _aggregate(samples)
    roundtrip_smoke = _roundtrip_smoke(
        inventory, args.bvh_cache, blender, settings
    )
    checks = _quality_checks(aggregate, settings)
    execution_passed = len(samples) == len(inventory) and roundtrip_smoke["passed"]
    quality_gate_passed = all(bool(item["passed"]) for item in checks)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "passed": execution_passed and quality_gate_passed,
        "execution_passed": execution_passed,
        "quality_gate_passed": quality_gate_passed,
        "codec": {"version": "SKAC v1", "quality": args.quality},
        "input": {
            "format": "fbx_via_blender_bvh",
            "families": list(FAMILIES),
            "file_count": len(inventory),
            "source_paths_in_report": False,
            "original_names_in_report": False,
            "assets_in_repository": False,
            "sampling": {
                "policy": "deterministic_center_window",
                "maximum_frames_per_motion": args.max_frames,
                "all_files_included": True,
            },
        },
        "backend": {
            "blender_version": version,
            "extraction_seconds_wall": extraction_seconds,
            "cache_hits": sum(bool(item["cache_hit"]) for item in extraction),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "extract_workers": args.extract_workers,
            "codec_workers": args.codec_workers,
            "codec_wall_seconds": codec_wall_seconds,
        },
        "aggregate": aggregate,
        "families": family_reports,
        "checks": checks,
        "fbx_roundtrip_smoke": roundtrip_smoke,
        "extraction": extraction,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    args.visual.write_text(_svg(report), encoding="utf-8", newline="\n")
    print(json.dumps({"passed": report["passed"], "execution_passed": execution_passed, "quality_gate_passed": quality_gate_passed, "checks": checks, "aggregate": aggregate}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
