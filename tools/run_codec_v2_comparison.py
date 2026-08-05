from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from skac_codec.bvh import read_bvh  # noqa: E402
from skac_codec.format import PREFIX, CodecSettings, decode_bytes, encode_bytes  # noqa: E402
from skac_codec.format_v2 import encode_v2_bytes, inspect_v2_bytes  # noqa: E402
from skac_codec.metrics import compression_metrics, roundtrip_metrics  # noqa: E402
from skac_codec.model import MotionClip  # noqa: E402
from skac_codec.retarget import _skeleton_height  # noqa: E402
from tools.run_codec_showcase import CHARACTERS, DEFAULT_SEED, select_animations  # noqa: E402


REPORT_SCHEMA = "skac.codec_version_comparison"
REPORT_VERSION = "1.0.0"
VERSION_LABELS = ("SKAC v1", "SKAC v2.1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SKAC v1 and v2.1 on the same public BVH sample."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--sample-report", type=Path)
    parser.add_argument("--animations-per-character", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--decode-iterations", type=int, default=3)
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--min-segment-frames", type=int, default=8)
    parser.add_argument("--max-segment-frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def _container_sections(data: bytes) -> dict[str, int]:
    fields = PREFIX.unpack_from(data)
    return {
        "format_major": int(fields[1]),
        "format_minor": int(fields[2]),
        "file_bytes": len(data),
        "metadata_bytes": int(fields[4]),
        "payload_bytes": int(fields[5]),
        "raw_payload_bytes": int(fields[6]),
    }


def _measure_decode(encoded: bytes, iterations: int) -> tuple[float, int]:
    if iterations < 1:
        raise ValueError("decode-iterations must be positive")
    decode_bytes(encoded)
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        decode_bytes(encoded)
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)

    tracemalloc.start()
    try:
        decode_bytes(encoded)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(statistics.median(durations)), int(peak_bytes)


def _measure_version(
    clip: MotionClip,
    *,
    encoder: Callable[[], bytes],
    decode_iterations: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    encoded = encoder()
    encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    decoded = decode_bytes(encoded)
    decode_ms, traced_peak_bytes = _measure_decode(encoded, decode_iterations)
    metrics = roundtrip_metrics(clip, decoded)
    height = _skeleton_height(clip.skeleton, 1)
    result: dict[str, Any] = {
        **_container_sections(encoded),
        **compression_metrics(clip, len(encoded)),
        **metrics,
        "encode_clip_ms": encode_ms,
        "decode_clip_ms_median": decode_ms,
        "decode_traced_peak_bytes": traced_peak_bytes,
        "global_position_error_max_height_fraction": (
            metrics["global_position_error_max"] / height
        ),
        "root_translation_error_max_height_fraction": (
            metrics["root_translation_error_max"] / height
        ),
    }
    if result["format_major"] == 2:
        inspected = inspect_v2_bytes(encoded)
        result["segment_count"] = int(inspected["codec"]["segment_count"])
        result["chunk_count"] = int(inspected["codec"]["chunk_count"])
    return result


def _aggregate(samples: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    versions = [sample["versions"][label] for sample in samples]
    playback_seconds = sum(float(sample["playback_budget_seconds"]) for sample in samples)
    encode_seconds = sum(float(item["encode_clip_ms"]) for item in versions) / 1000.0
    decode_seconds = sum(float(item["decode_clip_ms_median"]) for item in versions) / 1000.0
    raw_bytes = sum(int(item["raw_channel_bytes_float32"]) for item in versions)
    encoded_bytes = sum(int(item["file_bytes"]) for item in versions)
    joint_frames = sum(int(sample["joint_frames"]) for sample in samples)
    result: dict[str, Any] = {
        "sample_count": len(samples),
        "frame_count": sum(int(sample["frame_count"]) for sample in samples),
        "playback_budget_seconds": playback_seconds,
        "raw_channel_bytes_float32": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "metadata_bytes": sum(int(item["metadata_bytes"]) for item in versions),
        "payload_bytes": sum(int(item["payload_bytes"]) for item in versions),
        "raw_payload_bytes": sum(int(item["raw_payload_bytes"]) for item in versions),
        "compression_ratio_vs_float32_channels": raw_bytes / encoded_bytes,
        "bits_per_joint_per_frame": encoded_bytes * 8 / joint_frames,
        "encode_seconds": encode_seconds,
        "decode_seconds_median_sum": decode_seconds,
        "encode_realtime_factor": playback_seconds / encode_seconds,
        "decode_realtime_factor": playback_seconds / decode_seconds,
        "decode_traced_peak_bytes_max": max(
            int(item["decode_traced_peak_bytes"]) for item in versions
        ),
        "rotation_error_degrees_mean_weighted": float(
            np.average(
                [float(item["rotation_error_degrees_mean"]) for item in versions],
                weights=[int(sample["joint_frames"]) for sample in samples],
            )
        ),
        "rotation_error_degrees_max": max(
            float(item["rotation_error_degrees_max"]) for item in versions
        ),
        "global_position_error_max": max(
            float(item["global_position_error_max"]) for item in versions
        ),
        "global_position_error_max_height_fraction": max(
            float(item["global_position_error_max_height_fraction"]) for item in versions
        ),
        "root_translation_error_max_height_fraction": max(
            float(item["root_translation_error_max_height_fraction"]) for item in versions
        ),
    }
    if label == "SKAC v2.1":
        result["segment_count"] = sum(int(item["segment_count"]) for item in versions)
        result["chunk_count"] = sum(int(item["chunk_count"]) for item in versions)
    return result


def _load_sample_selection(path: Path) -> tuple[list[str], int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    sampling = report.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("sample report has no sampling object")
    animations = sampling.get("animations")
    seed = sampling.get("seed")
    if (
        not isinstance(animations, list)
        or not animations
        or any(not isinstance(item, str) or not item for item in animations)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("sample report has an invalid animation selection")
    return list(animations), seed


def _process_sample(
    job: tuple[str, str, str, int, CodecSettings, int, int]
) -> dict[str, Any]:
    (
        data_root,
        character,
        animation,
        decode_iterations,
        settings,
        min_segment_frames,
        max_segment_frames,
    ) = job
    clip = read_bvh(Path(data_root) / character / f"{animation}.bvh")
    versions = {
        "SKAC v1": _measure_version(
            clip,
            encoder=lambda: encode_bytes(clip, settings),
            decode_iterations=decode_iterations,
        ),
        "SKAC v2.1": _measure_version(
            clip,
            encoder=lambda: encode_v2_bytes(
                clip,
                settings,
                min_segment_frames=min_segment_frames,
                max_segment_frames=max_segment_frames,
            ),
            decode_iterations=decode_iterations,
        ),
    }
    return {
        "character": character,
        "animation": animation,
        "frame_count": clip.frame_count,
        "frame_time": clip.frame_time,
        "playback_budget_seconds": clip.frame_count * clip.frame_time,
        "joint_count": clip.skeleton.joint_count,
        "joint_frames": clip.frame_count * clip.skeleton.joint_count,
        "versions": versions,
    }


def run_comparison(
    data_root: Path,
    *,
    characters: Sequence[str] = CHARACTERS,
    animations_per_character: int = 20,
    seed: int = DEFAULT_SEED,
    selected_animations: Sequence[str] | None = None,
    decode_iterations: int = 3,
    settings: CodecSettings | None = None,
    min_segment_frames: int = 8,
    max_segment_frames: int = 32,
    workers: int = 1,
) -> dict[str, Any]:
    settings = settings or CodecSettings.preset("high")
    if selected_animations is None:
        selected = select_animations(
            data_root,
            characters=characters,
            count=animations_per_character,
            seed=seed,
        )
    else:
        selected = list(selected_animations)
        if len(selected) != animations_per_character or len(set(selected)) != len(selected):
            raise ValueError("selected animation count or uniqueness is invalid")
        for character in characters:
            missing = [
                animation
                for animation in selected
                if not (data_root / character / f"{animation}.bvh").is_file()
            ]
            if missing:
                raise ValueError(
                    f"public character {character} is missing {len(missing)} selected animations"
                )

    if workers < 1:
        raise ValueError("workers must be positive")
    jobs = [
        (
            str(data_root),
            character,
            animation,
            decode_iterations,
            settings,
            min_segment_frames,
            max_segment_frames,
        )
        for character in characters
        for animation in selected
    ]
    samples: list[dict[str, Any]]
    if workers == 1:
        samples = []
        for index, job in enumerate(jobs, 1):
            samples.append(_process_sample(job))
            print(
                f"completed paired sample {index}/{len(jobs)}",
                file=sys.stderr,
                flush=True,
            )
    else:
        ordered: list[dict[str, Any] | None] = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(_process_sample, job): index
                for index, job in enumerate(jobs)
            }
            completed_count = 0
            for future in as_completed(pending):
                ordered[pending[future]] = future.result()
                completed_count += 1
                print(
                    f"completed paired sample {completed_count}/{len(jobs)}",
                    file=sys.stderr,
                    flush=True,
                )
        if any(item is None for item in ordered):
            raise AssertionError("parallel comparison did not return every sample")
        samples = [item for item in ordered if item is not None]

    characters_report: list[dict[str, Any]] = []
    for character in characters:
        character_samples = [
            sample for sample in samples if sample["character"] == character
        ]
        character_versions = {
            label: _aggregate(character_samples, label) for label in VERSION_LABELS
        }
        characters_report.append(
            {"character": character, "versions": character_versions}
        )
        print(
            f"completed {character}: {len(character_samples)} paired animations",
            file=sys.stderr,
            flush=True,
        )

    overall = {label: _aggregate(samples, label) for label in VERSION_LABELS}
    v1 = overall["SKAC v1"]
    v2 = overall["SKAC v2.1"]
    saved = int(v1["encoded_bytes"]) - int(v2["encoded_bytes"])
    comparison = {
        "bytes_saved_v2_1_vs_v1": saved,
        "percent_smaller_v2_1_vs_v1": saved * 100.0 / int(v1["encoded_bytes"]),
        "clips_smaller_with_v2_1": sum(
            sample["versions"]["SKAC v2.1"]["file_bytes"]
            < sample["versions"]["SKAC v1"]["file_bytes"]
            for sample in samples
        ),
        "clips_equal_with_v2_1": sum(
            sample["versions"]["SKAC v2.1"]["file_bytes"]
            == sample["versions"]["SKAC v1"]["file_bytes"]
            for sample in samples
        ),
        "clips_larger_with_v2_1": sum(
            sample["versions"]["SKAC v2.1"]["file_bytes"]
            > sample["versions"]["SKAC v1"]["file_bytes"]
            for sample in samples
        ),
        "encode_seconds_delta_v2_1_minus_v1": (
            float(v2["encode_seconds"]) - float(v1["encode_seconds"])
        ),
        "decode_seconds_delta_v2_1_minus_v1": (
            float(v2["decode_seconds_median_sum"])
            - float(v1["decode_seconds_median_sum"])
        ),
    }
    expected_count = len(characters) * animations_per_character
    checks: list[dict[str, Any]] = [
        {
            "id": "paired_sample_count",
            "actual": len(samples),
            "comparator": "==",
            "limit": expected_count,
            "passed": len(samples) == expected_count,
        }
    ]
    for label in VERSION_LABELS:
        item = overall[label]
        checks.extend(
            [
                {
                    "id": f"{label.casefold().replace(' ', '_')}_rotation_error",
                    "actual": item["rotation_error_degrees_max"],
                    "comparator": "<=",
                    "limit": settings.rotation_error_degrees * 1.25,
                    "passed": item["rotation_error_degrees_max"]
                    <= settings.rotation_error_degrees * 1.25,
                },
                {
                    "id": f"{label.casefold().replace(' ', '_')}_root_translation_error",
                    "actual": item["root_translation_error_max_height_fraction"],
                    "comparator": "<=",
                    "limit": 0.0001,
                    "passed": item["root_translation_error_max_height_fraction"]
                    <= 0.0001,
                },
                {
                    "id": f"{label.casefold().replace(' ', '_')}_global_position_error",
                    "actual": item["global_position_error_max_height_fraction"],
                    "comparator": "<=",
                    "limit": 0.001,
                    "passed": item["global_position_error_max_height_fraction"]
                    <= 0.001,
                },
                {
                    "id": f"{label.casefold().replace(' ', '_')}_minimum_character_decode_realtime",
                    "actual": min(
                        float(character["versions"][label]["decode_realtime_factor"])
                        for character in characters_report
                    ),
                    "comparator": ">=",
                    "limit": 10.0,
                    "passed": min(
                        float(character["versions"][label]["decode_realtime_factor"])
                        for character in characters_report
                    )
                    >= 10.0,
                },
            ]
        )
    checks.append(
        {
            "id": "v2_1_size_not_larger_than_v1",
            "actual": v2["encoded_bytes"],
            "comparator": "<=",
            "limit": v1["encoded_bytes"],
            "passed": v2["encoded_bytes"] <= v1["encoded_bytes"],
        }
    )
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "passed": all(bool(item["passed"]) for item in checks),
        "scope": {
            "route": "same_character_codec_version_comparison",
            "source": "public_san_mixamo_test_split",
            "raw_size_baseline": "float32_animated_channels",
            "decode_timing": "whole_clip_median",
            "decode_memory": "python_tracemalloc_peak_per_whole_clip_decode",
            "io_included": False,
            "retargeting_included": False,
        },
        "sampling": {
            "seed": seed,
            "character_count": len(characters),
            "animations_per_character": animations_per_character,
            "sample_count": len(samples),
            "shared_animation_set": True,
            "animations": selected,
        },
        "codec": {
            "quality": settings.quality_name,
            "v2_min_segment_frames": min_segment_frames,
            "v2_max_segment_frames": max_segment_frames,
        },
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": sys.platform,
        },
        "decode_iterations": decode_iterations,
        "overall": overall,
        "comparison": comparison,
        "characters": characters_report,
        "checks": checks,
        "samples": samples,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    anchor: str | None = None,
) -> str:
    anchor_value = f' text-anchor="{anchor}"' if anchor else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}"'
        f'{anchor_value}>{escape(value)}</text>'
    )


def write_visual(path: Path, report: dict[str, Any]) -> None:
    width, height = 1400, 1040
    paper, ink, muted = "#f3f1ec", "#141414", "#696762"
    grid, accent, accent_soft, white = "#d0cec8", "#ff4f00", "#ffd8c7", "#ffffff"
    overall = report["overall"]
    v1, v2 = overall["SKAC v1"], overall["SKAC v2.1"]
    comparison = report["comparison"]
    mib = 1024.0 * 1024.0
    raw_mib = float(v1["raw_channel_bytes_float32"]) / mib
    v1_mib = float(v1["encoded_bytes"]) / mib
    v2_mib = float(v2["encoded_bytes"]) / mib
    max_character_bytes = max(
        float(character["versions"][label]["encoded_bytes"])
        for character in report["characters"]
        for label in VERSION_LABELS
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{paper}"/>',
        "<style>",
        "text{font-family:Inter,Segoe UI,Microsoft YaHei,Arial,sans-serif}",
        f".eyebrow{{font-size:12px;font-weight:750;fill:{muted};letter-spacing:2px}}",
        f".title{{font-size:38px;font-weight:800;fill:{ink};letter-spacing:-1px}}",
        f".subtitle{{font-size:14px;fill:{muted}}}",
        f".kpi{{font-size:33px;font-weight:800;fill:{ink}}}",
        f".accent{{font-size:33px;font-weight:800;fill:{accent}}}",
        f".kpilabel{{font-size:12px;font-weight:750;fill:{muted};letter-spacing:1px}}",
        f".panel{{font-size:19px;font-weight:750;fill:{ink}}}",
        f".label{{font-size:13px;font-weight:650;fill:{ink}}}",
        f".value{{font-size:12px;font-weight:750;fill:{ink}}}",
        f".valueaccent{{font-size:12px;font-weight:750;fill:{accent}}}",
        f".note{{font-size:11px;fill:{muted}}}",
        "</style>",
        _text(54, 40, "PUBLIC CODEC REGRESSION · SAME 160 CLIPS", css_class="eyebrow"),
        _text(54, 87, "SKAC v1 vs v2.1 — Public Codec Test", css_class="title"),
        _text(54, 116, "同一组 8 个公开角色 × 20 条动画 · high 档 · 不包含重定向", css_class="subtitle"),
        f'<line x1="54" y1="146" x2="1346" y2="146" stroke="{ink}" stroke-width="2"/>',
        f'<line x1="54" y1="146" x2="240" y2="146" stroke="{accent}" stroke-width="5"/>',
        _text(54, 184, "STORAGE / 总文件大小", css_class="kpilabel"),
        _text(54, 232, f"{raw_mib:.2f} MiB", css_class="kpi"),
        _text(252, 231, "→", css_class="panel"),
        _text(294, 232, f"{v1_mib:.2f} MiB", css_class="kpi"),
        _text(394, 232, "→", css_class="panel"),
        _text(436, 232, f"{v2_mib:.2f} MiB", css_class="accent"),
        _text(54, 261, "FLOAT32 channels → SKAC v1 → SKAC v2.1", css_class="subtitle"),
        f'<line x1="614" y1="174" x2="614" y2="274" stroke="{grid}"/>',
        _text(650, 184, "BYTES SAVED / 节省字节", css_class="kpilabel"),
        _text(650, 232, f'{int(comparison["bytes_saved_v2_1_vs_v1"]):+,} B', css_class="accent"),
        _text(650, 261, f'{float(comparison["percent_smaller_v2_1_vs_v1"]):.2f}% smaller vs v1', css_class="subtitle"),
        f'<line x1="946" y1="174" x2="946" y2="274" stroke="{grid}"/>',
        _text(982, 184, "MAX ROTATION ERROR / 最大旋转误差", css_class="kpilabel"),
        _text(982, 232, f'{float(v2["rotation_error_degrees_max"]):.5f}°', css_class="kpi"),
        _text(982, 261, "v2.1 · full local rotations", css_class="subtitle"),
        _text(54, 288, f'Strict global-position gate / 严格全局位置门槛 — v1: {float(v1["global_position_error_max_height_fraction"]):.6f} FAIL · v2.1: {float(v2["global_position_error_max_height_fraction"]):.6f} PASS · limit 0.001', css_class="subtitle"),
        f'<rect x="54" y="304" width="1292" height="86" fill="{white}"/>',
        _text(78, 335, "ENCODE TOTAL / 编码总耗时", css_class="kpilabel"),
        _text(78, 372, f'{float(v1["encode_seconds"]):.2f}s → {float(v2["encode_seconds"]):.2f}s', css_class="kpi"),
        _text(504, 335, "DECODE TOTAL / 解码中位耗时之和", css_class="kpilabel"),
        _text(504, 372, f'{float(v1["decode_seconds_median_sum"]):.2f}s → {float(v2["decode_seconds_median_sum"]):.2f}s', css_class="kpi"),
        _text(970, 335, "PEAK DECODE MEMORY / 单动画整段解码峰值", css_class="kpilabel"),
        _text(970, 372, f'{int(v1["decode_traced_peak_bytes_max"])/mib:.2f} → {int(v2["decode_traced_peak_bytes_max"])/mib:.2f} MiB', css_class="kpi"),
        _text(54, 432, "Stored bytes by character / 每个角色 20 条动画的文件字节数", css_class="panel"),
        _text(54, 457, "Gray: SKAC v1 · Orange: SKAC v2.1 · exact values shown at right", css_class="subtitle"),
    ]
    row_start, row_gap = 500.0, 56.0
    bar_x, bar_width = 220.0, 650.0
    for index, character in enumerate(report["characters"]):
        y = row_start + index * row_gap
        name = str(character["character"])
        old = character["versions"]["SKAC v1"]
        new = character["versions"]["SKAC v2.1"]
        old_width = float(old["encoded_bytes"]) * bar_width / max_character_bytes
        new_width = float(new["encoded_bytes"]) * bar_width / max_character_bytes
        lines.extend(
            [
                _text(198, y + 19, name, css_class="label", anchor="end"),
                f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="16" fill="{white}" stroke="{grid}"/>',
                f'<rect x="{bar_x}" y="{y}" width="{old_width:.1f}" height="7" fill="{ink}"/>',
                f'<rect x="{bar_x}" y="{y + 9}" width="{new_width:.1f}" height="7" fill="{accent}"/>',
                _text(900, y + 15, f'{int(old["encoded_bytes"]):,} B', css_class="value"),
                _text(1040, y + 15, "→", css_class="value"),
                _text(1080, y + 15, f'{int(new["encoded_bytes"]):,} B', css_class="valueaccent"),
                _text(1328, y + 15, f'{float(new["decode_realtime_factor"]):.1f}× decode', css_class="value", anchor="end"),
                f'<line x1="54" y1="{y + 37}" x2="1346" y2="{y + 37}" stroke="{grid}"/>',
            ]
        )
    lines.extend(
        [
            _text(54, 955, f'{int(comparison["clips_larger_with_v2_1"])}/{int(report["sampling"]["sample_count"])} clips are larger in v2.1; no per-clip fallback is applied. / v2.1 有部分动画更大，当前未启用逐动画回退。', css_class="note"),
            _text(54, 978, "Timing excludes file I/O. Peak memory is Python tracemalloc evidence, not process RSS. Full per-clip data is in JSON.", css_class="note"),
            _text(54, 1001, "计时不含文件读写；峰值内存来自 Python tracemalloc，不等于进程 RSS。逐动画数据见 JSON。", css_class="note"),
            _text(54, 1024, "Same-character Codec reconstruction only; no cross-skeleton quality claim is made.", css_class="note"),
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected: Sequence[str] | None = None
    seed = args.seed
    if args.sample_report is not None:
        selected, seed = _load_sample_selection(args.sample_report)
        if len(selected) != args.animations_per_character:
            raise ValueError(
                "sample report animation count differs from --animations-per-character"
            )
    report = run_comparison(
        args.data_root.resolve(),
        animations_per_character=args.animations_per_character,
        seed=seed,
        selected_animations=selected,
        decode_iterations=args.decode_iterations,
        settings=CodecSettings.preset(args.quality),
        min_segment_frames=args.min_segment_frames,
        max_segment_frames=args.max_segment_frames,
        workers=args.workers,
    )
    write_report(args.output, report)
    write_visual(args.visual, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "sample_count": report["sampling"]["sample_count"],
                "output": str(args.output),
                "visual": str(args.visual),
                "overall": report["overall"],
                "comparison": report["comparison"],
                "failed_checks": [
                    item["id"] for item in report["checks"] if not item["passed"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
