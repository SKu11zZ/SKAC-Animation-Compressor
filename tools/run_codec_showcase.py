from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from skac_codec.bvh import read_bvh  # noqa: E402
from skac_codec.format import CodecSettings, decode_bytes, encode_bytes  # noqa: E402
from skac_codec.math3d import quaternion_angular_error_degrees  # noqa: E402
from skac_codec.metrics import compression_metrics  # noqa: E402
from skac_codec.retarget import _skeleton_height  # noqa: E402


CHARACTERS = (
    "Aj",
    "BigVegas",
    "Goblin_m",
    "Kaya",
    "Mousey_m",
    "Mremireh_m",
    "SportyGranny",
    "Vampire_m",
)
REPORT_SCHEMA = "skac.codec_showcase"
REPORT_VERSION = "1.0.0"
DEFAULT_SEED = 20260801


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark same-character SKAC compression and whole-clip decode on a "
            "deterministic public animation sample."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument(
        "--quality", choices=("low", "medium", "high"), default="high"
    )
    parser.add_argument("--animations-per-character", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--decode-iterations", type=int, default=5)
    return parser.parse_args(argv)


def select_animations(
    data_root: Path,
    *,
    characters: Sequence[str],
    count: int,
    seed: int,
) -> list[str]:
    if count < 1:
        raise ValueError("animations-per-character must be positive")
    common: set[str] | None = None
    for character in characters:
        character_root = data_root / character
        if not character_root.is_dir():
            raise ValueError(f"missing public character directory: {character}")
        names = {path.stem for path in character_root.glob("*.bvh") if path.is_file()}
        common = names if common is None else common & names
    available = sorted(common or set())
    if len(available) < count:
        raise ValueError(
            f"only {len(available)} common animations are available; {count} requested"
        )
    return sorted(random.Random(seed).sample(available, count))


def _measure_decode_ms(encoded: bytes, iterations: int) -> float:
    if iterations < 1:
        raise ValueError("decode-iterations must be positive")
    decode_bytes(encoded)
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        decode_bytes(encoded)
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return float(statistics.median(durations))


def _aggregate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_bytes = sum(int(item["raw_channel_bytes_float32"]) for item in samples)
    encoded_bytes = sum(int(item["encoded_bytes"]) for item in samples)
    playback_seconds = sum(float(item["playback_budget_seconds"]) for item in samples)
    decode_seconds = sum(float(item["decode_clip_ms_median"]) for item in samples) / 1000.0
    encode_seconds = sum(float(item["encode_clip_ms"]) for item in samples) / 1000.0
    joint_frames = sum(int(item["joint_frames"]) for item in samples)
    return {
        "sample_count": len(samples),
        "frame_count": sum(int(item["frame_count"]) for item in samples),
        "playback_budget_seconds": playback_seconds,
        "raw_channel_bytes_float32": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "compression_ratio_vs_float32_channels": raw_bytes / encoded_bytes,
        "bits_per_joint_per_frame": encoded_bytes * 8 / joint_frames,
        "encode_realtime_factor": playback_seconds / encode_seconds,
        "decode_realtime_factor": playback_seconds / decode_seconds,
        "rotation_error_degrees_max": max(
            float(item["rotation_error_degrees_max"]) for item in samples
        ),
        "root_translation_error_max_height_fraction": max(
            float(item["root_translation_error_max_height_fraction"])
            for item in samples
        ),
    }


def run_benchmark(
    data_root: Path,
    *,
    characters: Sequence[str] = CHARACTERS,
    animations_per_character: int = 20,
    seed: int = DEFAULT_SEED,
    decode_iterations: int = 5,
    settings: CodecSettings | None = None,
) -> dict[str, Any]:
    settings = settings or CodecSettings.preset("high")
    selected = select_animations(
        data_root,
        characters=characters,
        count=animations_per_character,
        seed=seed,
    )
    samples: list[dict[str, Any]] = []
    character_reports: list[dict[str, Any]] = []
    for character in characters:
        character_samples: list[dict[str, Any]] = []
        for animation in selected:
            clip = read_bvh(data_root / character / f"{animation}.bvh")
            encode_started = time.perf_counter_ns()
            encoded = encode_bytes(clip, settings)
            encode_ms = (time.perf_counter_ns() - encode_started) / 1_000_000.0
            decoded = decode_bytes(encoded)
            rotation_error = quaternion_angular_error_degrees(
                clip.local_rotations, decoded.local_rotations
            )
            translation_error = np.linalg.norm(
                clip.local_translations - decoded.local_translations, axis=-1
            )
            root_translation_error = translation_error[:, 0]
            codec = {
                **compression_metrics(clip, len(encoded)),
                "rotation_error_degrees_mean": float(np.mean(rotation_error)),
                "rotation_error_degrees_max": float(np.max(rotation_error)),
                "translation_error_mean": float(np.mean(translation_error)),
                "translation_error_max": float(np.max(translation_error)),
                "root_translation_error_mean": float(
                    np.mean(root_translation_error)
                ),
                "root_translation_error_max": float(np.max(root_translation_error)),
            }
            height = _skeleton_height(clip.skeleton, 1)
            record = {
                "character": character,
                "animation": animation,
                "frame_count": clip.frame_count,
                "frame_time": clip.frame_time,
                "playback_budget_seconds": clip.frame_count * clip.frame_time,
                "joint_count": clip.skeleton.joint_count,
                "joint_frames": clip.frame_count * clip.skeleton.joint_count,
                "encode_clip_ms": encode_ms,
                "decode_clip_ms_median": _measure_decode_ms(
                    encoded, decode_iterations
                ),
                **codec,
                "root_translation_error_max_height_fraction": (
                    codec["root_translation_error_max"] / height
                ),
            }
            samples.append(record)
            character_samples.append(record)
        character_reports.append(
            {"character": character, **_aggregate(character_samples)}
        )
        print(
            f"completed {character}: {len(character_samples)} animations",
            file=sys.stderr,
            flush=True,
        )

    overall = _aggregate(samples)
    checks = [
        {
            "id": "sample_count",
            "actual": len(samples),
            "comparator": "==",
            "limit": len(characters) * animations_per_character,
            "passed": len(samples) == len(characters) * animations_per_character,
        },
        {
            "id": "rotation_error_degrees_max",
            "actual": overall["rotation_error_degrees_max"],
            "comparator": "<=",
            "limit": settings.rotation_error_degrees * 1.25,
            "passed": overall["rotation_error_degrees_max"]
            <= settings.rotation_error_degrees * 1.25,
        },
        {
            "id": "root_translation_error_max_height_fraction",
            "actual": overall["root_translation_error_max_height_fraction"],
            "comparator": "<=",
            "limit": 0.0001,
            "passed": overall["root_translation_error_max_height_fraction"]
            <= 0.0001,
        },
        {
            "id": "minimum_character_decode_realtime_factor",
            "actual": min(
                float(item["decode_realtime_factor"]) for item in character_reports
            ),
            "comparator": ">=",
            "limit": 10.0,
            "passed": min(
                float(item["decode_realtime_factor"]) for item in character_reports
            )
            >= 10.0,
        },
    ]
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "passed": all(item["passed"] for item in checks),
        "scope": {
            "route": "same_character_codec_roundtrip",
            "source": "public_san_mixamo_test_split",
            "raw_size_baseline": "float32_animated_channels",
            "decode_timing": "whole_clip_median",
            "io_included": False,
            "reconstruction_checks": "full_local_rotation_and_root_translation_sequences",
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
            "rotation_bits": settings.rotation_bits,
            "translation_bits": settings.translation_bits,
            "rotation_error_degrees": settings.rotation_error_degrees,
            "translation_error_fraction": settings.translation_error_fraction,
            "zlib_level": settings.zlib_level,
        },
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": sys.platform,
        },
        "decode_iterations": decode_iterations,
        "overall": overall,
        "characters": character_reports,
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
    anchor_attribute = f' text-anchor="{anchor}"' if anchor else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}"'
        f'{anchor_attribute}>{escape(value)}</text>'
    )


def write_visual(path: Path, report: dict[str, Any]) -> None:
    """Write the concrete-value, bilingual public benchmark graphic."""
    width, height = 1400, 960
    paper = "#f4f2ed"
    ink = "#151515"
    muted = "#686763"
    grid = "#d2d0ca"
    accent = "#ff4f00"
    accent_open = "#ffd8c7"
    white = "#ffffff"
    characters = report["characters"]
    overall = report["overall"]
    sample_count = int(report["sampling"]["sample_count"])
    animation_count = int(report["sampling"]["animations_per_character"])
    decode_iterations = int(report["decode_iterations"])
    mib = 1024.0 * 1024.0
    raw_mib = float(overall["raw_channel_bytes_float32"]) / mib
    encoded_mib = float(overall["encoded_bytes"]) / mib
    saved_mib = raw_mib - encoded_mib
    playback_seconds = float(overall["playback_budget_seconds"])
    encode_seconds = playback_seconds / float(overall["encode_realtime_factor"])
    decode_seconds = playback_seconds / float(overall["decode_realtime_factor"])
    raw_max = max(
        1.0,
        max(float(item["raw_channel_bytes_float32"]) / mib for item in characters)
        * 1.08,
    )
    encode_max = max(
        1.0,
        max(
            float(item["playback_budget_seconds"])
            / float(item["encode_realtime_factor"])
            for item in characters
        )
        * 1.08,
    )
    decode_max_ms = max(
        1.0,
        max(
            1000.0
            * float(item["playback_budget_seconds"])
            / float(item["decode_realtime_factor"])
            for item in characters
        )
        * 1.08,
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{paper}"/>',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif}",
        f".eyebrow{{font-size:12px;font-weight:750;fill:{muted};letter-spacing:2px}}",
        f".title{{font-size:39px;font-weight:800;fill:{ink};letter-spacing:-1px}}",
        f".subtitle{{font-size:14px;fill:{muted}}}",
        f".kpi{{font-size:36px;font-weight:800;fill:{ink};letter-spacing:-1px}}",
        f".kpiaccent{{font-size:36px;font-weight:800;fill:{accent};letter-spacing:-1px}}",
        f".kpilabel{{font-size:12px;font-weight:750;fill:{muted};letter-spacing:1.1px}}",
        f".panel{{font-size:19px;font-weight:750;fill:{ink}}}",
        f".label{{font-size:13px;font-weight:650;fill:{ink}}}",
        f".value{{font-size:12px;font-weight:750;fill:{ink}}}",
        f".valueaccent{{font-size:12px;font-weight:750;fill:{accent}}}",
        f".axis{{font-size:11px;fill:{muted}}}",
        f".note{{font-size:11px;fill:{muted}}}",
        "</style>",
        _text(54, 40, "PUBLIC BENCHMARK · FIXED SEED 20260801", css_class="eyebrow"),
        _text(54, 88, "SKAC Codec — 160-Clip Public Run", css_class="title"),
        _text(
            54,
            118,
            f"8 public characters · {animation_count} shared clips each · {sample_count} BVH files · "
            f"{playback_seconds:.2f} seconds of motion · high preset",
            css_class="subtitle",
        ),
        _text(
            54,
            142,
            f"8 个公开角色 · 每个角色 {animation_count} 条相同动画 · 共 {sample_count} 条 BVH · "
            f"动画总时长 {playback_seconds:.2f} 秒 · high 档",
            css_class="subtitle",
        ),
        f'<line x1="54" y1="166" x2="1346" y2="166" stroke="{ink}" stroke-width="2"/>',
        f'<line x1="54" y1="166" x2="226" y2="166" stroke="{accent}" stroke-width="5"/>',
        _text(54, 201, "STORAGE / 存储", css_class="kpilabel"),
        _text(54, 250, f"{raw_mib:.2f} MiB", css_class="kpi"),
        _text(265, 249, "→", css_class="panel"),
        _text(310, 250, f"{encoded_mib:.2f} MiB", css_class="kpiaccent"),
        _text(
            54,
            278,
            f"Float32 channels → .skac bytes · {saved_mib:.2f} MiB removed",
            css_class="subtitle",
        ),
        f'<line x1="568" y1="192" x2="568" y2="282" stroke="{grid}"/>',
        _text(606, 201, "ENCODE / 编码", css_class="kpilabel"),
        _text(606, 250, f"{encode_seconds:.2f} s", css_class="kpi"),
        _text(606, 278, f"{sample_count} clips total", css_class="subtitle"),
        f'<line x1="837" y1="192" x2="837" y2="282" stroke="{grid}"/>',
        _text(875, 201, "DECODE / 解码", css_class="kpilabel"),
        _text(875, 250, f"{decode_seconds:.2f} s", css_class="kpiaccent"),
        _text(
            875,
            278,
            f"whole-clip median · {decode_iterations} runs",
            css_class="subtitle",
        ),
        f'<line x1="1114" y1="192" x2="1114" y2="282" stroke="{grid}"/>',
        _text(1152, 201, "MAX ERROR / 最大误差", css_class="kpilabel"),
        _text(
            1152,
            250,
            f'{overall["rotation_error_degrees_max"]:.5f}°',
            css_class="kpi",
        ),
        _text(1152, 278, "local rotation", css_class="subtitle"),
        f'<rect x="54" y="310" width="1292" height="38" fill="{white}"/>',
        _text(
            72,
            335,
            "READ IT AS: exact stored bytes on the left; exact measured processing time on the right.",
            css_class="label",
        ),
        _text(
            1328,
            335,
            "左边看体积，右边看实测耗时",
            css_class="label",
            anchor="end",
        ),
    ]

    panel_y = 390
    lines.extend(
        [
            _text(54, panel_y, "Stored bytes by character / 每个角色实际占用", css_class="panel"),
            _text(
                54,
                panel_y + 24,
                "Twenty clips combined · both bars share the same MiB scale",
                css_class="subtitle",
            ),
            _text(
                54,
                panel_y + 44,
                "每个角色合计 20 条动画 · 两条柱使用同一 MiB 刻度",
                css_class="subtitle",
            ),
            _text(744, panel_y, "Measured processing time / 实测处理耗时", css_class="panel"),
            _text(
                744,
                panel_y + 24,
                "Exact totals for twenty clips · file I/O excluded",
                css_class="subtitle",
            ),
            _text(
                744,
                panel_y + 44,
                "每个角色 20 条动画的累计耗时 · 不含文件读写",
                css_class="subtitle",
            ),
        ]
    )
    left_plot_x, left_plot_width = 170.0, 310.0
    encode_plot_x, encode_plot_width = 842.0, 142.0
    decode_plot_x, decode_plot_width = 1132.0, 142.0
    row_start, row_gap = 468.0, 54.0
    lines.extend(
        [
            f'<rect x="54" y="438" width="12" height="12" fill="{ink}"/>',
            _text(74, 448, "FLOAT32", css_class="axis"),
            f'<rect x="137" y="438" width="12" height="12" fill="{accent}"/>',
            _text(157, 448, "SKAC", css_class="axis"),
            _text(688, 448, "RAW → SKAC", css_class="axis", anchor="end"),
            _text(842, 448, "ENCODE · SECONDS", css_class="axis"),
            _text(1132, 448, "DECODE · MILLISECONDS", css_class="axis"),
        ]
    )
    for index, item in enumerate(characters):
        y = row_start + index * row_gap
        character = str(item["character"])
        item_raw_mib = float(item["raw_channel_bytes_float32"]) / mib
        item_encoded_mib = float(item["encoded_bytes"]) / mib
        item_encode_seconds = (
            float(item["playback_budget_seconds"])
            / float(item["encode_realtime_factor"])
        )
        item_decode_ms = (
            1000.0
            * float(item["playback_budget_seconds"])
            / float(item["decode_realtime_factor"])
        )
        raw_width = left_plot_width * item_raw_mib / raw_max
        encoded_width = left_plot_width * item_encoded_mib / raw_max
        encode_width = encode_plot_width * item_encode_seconds / encode_max
        decode_width = decode_plot_width * item_decode_ms / decode_max_ms
        lines.extend(
            [
                f'<line x1="54" y1="{y + 37:.1f}" x2="1346" y2="{y + 37:.1f}" stroke="{grid}"/>',
                _text(
                    left_plot_x - 14,
                    y + 15,
                    character,
                    css_class="label",
                    anchor="end",
                ),
                f'<rect x="{left_plot_x:.1f}" y="{y:.1f}" width="{raw_width:.1f}" height="8" fill="{ink}"/>',
                f'<rect x="{left_plot_x:.1f}" y="{y + 11:.1f}" width="{encoded_width:.1f}" height="8" fill="{accent}"/>',
                _text(
                    688,
                    y + 15,
                    f"{item_raw_mib:.2f} → {item_encoded_mib:.2f} MiB",
                    css_class="value",
                    anchor="end",
                ),
                _text(812, y + 15, character, css_class="label", anchor="end"),
                f'<rect x="{encode_plot_x:.1f}" y="{y + 2:.1f}" width="{encode_plot_width:.1f}" height="14" fill="{white}" stroke="{grid}"/>',
                f'<rect x="{encode_plot_x:.1f}" y="{y + 2:.1f}" width="{encode_width:.1f}" height="14" fill="{ink}"/>',
                _text(994, y + 15, f"{item_encode_seconds:.2f} s", css_class="value"),
                f'<rect x="{decode_plot_x:.1f}" y="{y + 2:.1f}" width="{decode_plot_width:.1f}" height="14" fill="{accent_open}" stroke="{grid}"/>',
                f'<rect x="{decode_plot_x:.1f}" y="{y + 2:.1f}" width="{decode_width:.1f}" height="14" fill="{accent}"/>',
                _text(
                    1284,
                    y + 15,
                    f"{item_decode_ms:,.0f} ms",
                    css_class="valueaccent",
                ),
            ]
        )
    lines.extend(
        [
            _text(
                54,
                920,
                f"Decode value is the sum of per-clip medians across {decode_iterations} runs. "
                "Full local rotations and root translation are checked. Timing is machine-dependent.",
                css_class="note",
            ),
            _text(
                54,
                942,
                "解码值为每条动画重复测量后的中位数之和；完整检查局部旋转与根位移。计时结果与机器有关，逐条数据见 JSON。",
                css_class="note",
            ),
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        args.data_root.resolve(),
        animations_per_character=args.animations_per_character,
        seed=args.seed,
        decode_iterations=args.decode_iterations,
        settings=CodecSettings.preset(args.quality),
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
