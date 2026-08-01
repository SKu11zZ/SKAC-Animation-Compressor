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
    width, height = 1400, 900
    ink = "#17202a"
    muted = "#667085"
    grid = "#d7dee7"
    blue = "#2864dc"
    blue_open = "#dbe8ff"
    gold = "#c98a16"
    gold_open = "#f8e7bd"
    surface = "#f7f9fc"
    characters = report["characters"]
    overall = report["overall"]
    sample_count = int(report["sampling"]["sample_count"])
    animation_count = int(report["sampling"]["animations_per_character"])
    decode_iterations = int(report["decode_iterations"])
    compression_max = max(
        5.0,
        max(float(item["compression_ratio_vs_float32_channels"]) for item in characters)
        * 1.15,
    )
    decode_max = max(
        20.0,
        max(float(item["decode_realtime_factor"]) for item in characters) * 1.15,
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif}",
        f".title{{font-size:34px;font-weight:750;fill:{ink}}}",
        f".subtitle{{font-size:15px;fill:{muted}}}",
        f".kpi{{font-size:30px;font-weight:750;fill:{ink}}}",
        f".kpilabel{{font-size:13px;font-weight:650;fill:{muted};letter-spacing:.5px}}",
        f".panel{{font-size:18px;font-weight:700;fill:{ink}}}",
        f".label{{font-size:14px;font-weight:600;fill:{ink}}}",
        f".value{{font-size:13px;font-weight:700;fill:{ink}}}",
        f".axis{{font-size:12px;fill:{muted}}}",
        f".note{{font-size:12px;fill:{muted}}}",
        "</style>",
        _text(54, 58, "SKAC Codec Performance", css_class="title"),
        _text(
            54,
            88,
            f"8 public characters x {animation_count} shared random animations | "
            f"{sample_count} BVH clips | high quality | whole-clip decode",
            css_class="subtitle",
        ),
    ]

    cards = [
        (
            f'{overall["compression_ratio_vs_float32_channels"]:.2f}x',
            "SMALLER THAN FLOAT32 CHANNELS",
        ),
        (f'{overall["encode_realtime_factor"]:.2f}x', "OFFLINE ENCODE REALTIME"),
        (f'{overall["decode_realtime_factor"]:.1f}x', "WHOLE-CLIP DECODE REALTIME"),
        (f'{overall["rotation_error_degrees_max"]:.4f} deg', "MAX ROTATION ERROR"),
    ]
    card_y, card_height = 122, 112
    card_width = 306
    for index, (value, label) in enumerate(cards):
        x = 54 + index * 332
        lines.append(
            f'<rect x="{x}" y="{card_y}" width="{card_width}" height="{card_height}" '
            f'rx="10" fill="{surface}" stroke="{grid}"/>'
        )
        lines.append(_text(x + 22, card_y + 48, value, css_class="kpi"))
        lines.append(_text(x + 22, card_y + 80, label, css_class="kpilabel"))

    panel_y = 286
    lines.extend(
        [
            _text(54, panel_y, "Compression ratio by character", css_class="panel"),
            _text(
                54,
                panel_y + 24,
                "Aggregate raw float32 channel bytes / encoded bytes; higher is better",
                css_class="subtitle",
            ),
            _text(746, panel_y, "Whole-clip decode speed by character", css_class="panel"),
            _text(
                746,
                panel_y + 24,
                f"Aggregate animation duration / median decode time ({decode_iterations} runs per clip)",
                css_class="subtitle",
            ),
        ]
    )
    left_plot_x, right_plot_x = 190.0, 882.0
    plot_width = 430.0
    row_start, row_gap, bar_height = 342.0, 57.0, 24.0
    for tick in range(6):
        left_value = compression_max * tick / 5
        right_value = decode_max * tick / 5
        left_x = left_plot_x + plot_width * tick / 5
        right_x = right_plot_x + plot_width * tick / 5
        lines.extend(
            [
                f'<line x1="{left_x:.1f}" y1="326" x2="{left_x:.1f}" y2="807" stroke="{grid}"/>',
                f'<line x1="{right_x:.1f}" y1="326" x2="{right_x:.1f}" y2="807" stroke="{grid}"/>',
                _text(left_x, 829, f"{left_value:.1f}x", css_class="axis", anchor="middle"),
                _text(right_x, 829, f"{right_value:.0f}x", css_class="axis", anchor="middle"),
            ]
        )
    gate_x = right_plot_x + plot_width * 10.0 / decode_max
    lines.extend(
        [
            f'<line x1="{gate_x:.1f}" y1="326" x2="{gate_x:.1f}" y2="807" stroke="{ink}" stroke-width="2" stroke-dasharray="5 5"/>',
            _text(gate_x + 6, 338, "10x gate", css_class="axis"),
        ]
    )
    for index, item in enumerate(characters):
        y = row_start + index * row_gap
        character = str(item["character"])
        compression = float(item["compression_ratio_vs_float32_channels"])
        decode = float(item["decode_realtime_factor"])
        compression_width = plot_width * compression / compression_max
        decode_width = plot_width * decode / decode_max
        lines.extend(
            [
                _text(left_plot_x - 14, y + 17, character, css_class="label", anchor="end"),
                f'<rect x="{left_plot_x:.1f}" y="{y:.1f}" width="{plot_width:.1f}" height="{bar_height}" rx="4" fill="{blue_open}"/>',
                f'<rect x="{left_plot_x:.1f}" y="{y:.1f}" width="{compression_width:.1f}" height="{bar_height}" rx="4" fill="{blue}"/>',
                _text(
                    left_plot_x + compression_width + 8,
                    y + 17,
                    f"{compression:.2f}x",
                    css_class="value",
                ),
                _text(right_plot_x - 14, y + 17, character, css_class="label", anchor="end"),
                f'<rect x="{right_plot_x:.1f}" y="{y:.1f}" width="{plot_width:.1f}" height="{bar_height}" rx="4" fill="{gold_open}"/>',
                f'<rect x="{right_plot_x:.1f}" y="{y:.1f}" width="{decode_width:.1f}" height="{bar_height}" rx="4" fill="{gold}"/>',
                _text(
                    right_plot_x + decode_width + 8,
                    y + 17,
                    f"{decode:.1f}x",
                    css_class="value",
                ),
            ]
        )
    lines.extend(
        [
            _text(
                54,
                872,
                "Fixed seed 20260801. Same 20 animation IDs for every character. File I/O excluded. "
                "Full local rotations and root translation are checked; see JSON for per-clip results.",
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
