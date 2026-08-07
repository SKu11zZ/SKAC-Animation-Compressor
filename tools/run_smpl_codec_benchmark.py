from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from skac_codec.format import CodecSettings, decode_bytes, encode_bytes  # noqa: E402
from skac_codec.metrics import compression_metrics, roundtrip_metrics  # noqa: E402
from skac_codec.model import MotionClip  # noqa: E402
from skac_codec.smpl import read_smpl_npz  # noqa: E402


SCHEMA = "skac.smplx_codec_benchmark"
SCHEMA_VERSION = "1.0.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SKAC v1 on a neutral local SMPL-X motion cache."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=512)
    return parser.parse_args(argv)


def _motion_inventory(root: Path) -> tuple[list[tuple[str, Path, int]], list[str]]:
    motions: list[tuple[str, Path, int]] = []
    skipped: list[str] = []
    for path in sorted(root.glob("smpl_*.npz")):
        identifier = path.stem
        try:
            with np.load(path, allow_pickle=False) as archive:
                poses = archive["poses"]
                translations = archive["trans"]
                rate = archive["mocap_frame_rate"]
                valid = (
                    poses.ndim == 2
                    and poses.shape[1] == 165
                    and translations.shape == (poses.shape[0], 3)
                    and np.asarray(rate).size == 1
                )
        except (KeyError, ValueError, OSError):
            valid = False
        if valid:
            motions.append((identifier, path, int(poses.shape[0])))
        else:
            skipped.append(identifier)
    if not motions:
        raise ValueError("the supplied cache contains no valid SMPL-X motion streams")
    return motions, skipped


def _measure(job: tuple[str, str, int, CodecSettings]) -> dict[str, Any]:
    identifier, source, max_frames, settings = job
    full_clip = read_smpl_npz(Path(source))
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
    started = time.perf_counter_ns()
    decoded = decode_bytes(encoded)
    decode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "id": identifier,
        "source_frame_count": full_clip.frame_count,
        "window_start_frame": start,
        "frame_count": clip.frame_count,
        "joint_count": clip.skeleton.joint_count,
        "duration_seconds": clip.duration_seconds,
        "encode_clip_ms": encode_ms,
        "decode_clip_ms": decode_ms,
        **compression_metrics(clip, len(encoded)),
        **roundtrip_metrics(clip, decoded),
    }


def _weighted_mean(samples: list[dict[str, Any]], field: str) -> float:
    return float(
        np.average(
            [float(item[field]) for item in samples],
            weights=[int(item["frame_count"]) * int(item["joint_count"]) for item in samples],
        )
    )


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    raw_bytes = sum(int(item["raw_channel_bytes_float32"]) for item in samples)
    encoded_bytes = sum(int(item["encoded_bytes"]) for item in samples)
    duration = sum(float(item["duration_seconds"]) for item in samples)
    encode_seconds = sum(float(item["encode_clip_ms"]) for item in samples) / 1000.0
    decode_seconds = sum(float(item["decode_clip_ms"]) for item in samples) / 1000.0
    bytes_per_frame = [
        float(item["encoded_bytes"]) / int(item["frame_count"]) for item in samples
    ]
    return {
        "motion_count": len(samples),
        "frame_count": sum(int(item["frame_count"]) for item in samples),
        "joint_count": 55,
        "playback_seconds": duration,
        "raw_channel_bytes_float32": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "compression_ratio_vs_float32_channels": raw_bytes / encoded_bytes,
        "encoded_bytes_per_frame_median": statistics.median(bytes_per_frame),
        "encoded_bytes_per_frame_p95": float(np.percentile(bytes_per_frame, 95)),
        "encode_seconds_sum": encode_seconds,
        "decode_seconds_sum": decode_seconds,
        "encode_realtime_factor": duration / encode_seconds,
        "decode_realtime_factor": duration / decode_seconds,
        "rotation_error_degrees_mean_weighted": _weighted_mean(
            samples, "rotation_error_degrees_mean"
        ),
        "rotation_error_degrees_max": max(
            float(item["rotation_error_degrees_max"]) for item in samples
        ),
        "root_translation_error_mean": _weighted_mean(
            samples, "root_translation_error_mean"
        ),
        "root_translation_error_max": max(
            float(item["root_translation_error_max"]) for item in samples
        ),
        "canonical_proxy_global_position_error_max": max(
            float(item["global_position_error_max"]) for item in samples
        ),
    }


def _fmt_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):,.2f} MiB"


def _svg(report: dict[str, Any]) -> str:
    a = report["aggregate"]
    cards = [
        ("MOTIONS / 动画", f'{a["motion_count"]:,}'),
        ("FRAMES / 帧", f'{a["frame_count"]:,}'),
        ("RAW CHANNELS / 原始通道", _fmt_bytes(a["raw_channel_bytes_float32"])),
        ("SKAC FILES / 压缩后", _fmt_bytes(a["encoded_bytes"])),
        ("COMPRESSION / 压缩比", f'{a["compression_ratio_vs_float32_channels"]:.2f}×'),
        ("MAX ROTATION ERROR / 最大旋转误差", f'{a["rotation_error_degrees_max"]:.4f}°'),
    ]
    card_svg = []
    for index, (label, value) in enumerate(cards):
        col, row = index % 3, index // 3
        x, y = 54 + col * 398, 176 + row * 132
        card_svg.append(
            f'<rect x="{x}" y="{y}" width="370" height="104" rx="12" class="card"/>'
            f'<text x="{x + 22}" y="{y + 31}" class="label">{escape(label)}</text>'
            f'<text x="{x + 22}" y="{y + 75}" class="value">{escape(value)}</text>'
        )
    raw = int(a["raw_channel_bytes_float32"])
    encoded = int(a["encoded_bytes"])
    encoded_width = max(8.0, 1030.0 * encoded / raw)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<style>
  .bg{{fill:#0e1012}} .card{{fill:#171a1e;stroke:#30343a}} .title{{fill:#f3f4f6;font:700 34px Arial,sans-serif}}
  .sub{{fill:#979da7;font:15px Arial,sans-serif}} .label{{fill:#9da3ad;font:700 12px Arial,sans-serif;letter-spacing:.5px}}
  .value{{fill:#f6f7f8;font:700 28px Arial,sans-serif}} .axis{{fill:#a9afb8;font:13px Arial,sans-serif}}
  .note{{fill:#858c96;font:12px Arial,sans-serif}} .raw{{fill:#e9eaec}} .accent{{fill:#25e6c8}}
</style>
<rect width="1280" height="720" class="bg"/>
<rect x="0" y="0" width="12" height="720" class="accent"/>
<text x="54" y="65" class="title">SKAC v1 · SMPL-X Parameter Stream Benchmark</text>
<text x="54" y="96" class="sub">{a["motion_count"]} public motions · {a["frame_count"]:,} evaluated / {report["input"]["source_frame_count"]:,} source frames · 120 FPS</text>
<text x="54" y="123" class="sub">Rotation and root translation are exact-source metrics; global position is a canonical-topology proxy.</text>
<text x="54" y="145" class="sub">旋转与根位移按原始参数计算；全局位置仅为标准化拓扑代理，不代表人体模型或 Mesh 误差。</text>
{''.join(card_svg)}
<text x="54" y="478" class="label">TOTAL STORAGE / 总存储量</text>
<rect x="190" y="508" width="1030" height="28" rx="6" class="raw"/>
<rect x="190" y="560" width="{encoded_width:.1f}" height="28" rx="6" class="accent"/>
<text x="54" y="528" class="axis">Float32</text><text x="54" y="580" class="axis">SKAC v1</text>
<text x="1208" y="528" text-anchor="end" class="axis">{escape(_fmt_bytes(raw))}</text>
<text x="{190 + encoded_width + 10:.1f}" y="580" class="axis">{escape(_fmt_bytes(encoded))}</text>
<text x="54" y="642" class="note">Summed clip encode: {a["encode_seconds_sum"]:.2f} s ({a["encode_realtime_factor"]:.2f}× realtime) · summed Python decode: {a["decode_seconds_sum"]:.2f} s ({a["decode_realtime_factor"]:.1f}× realtime)</text>
<text x="54" y="668" class="note">全片编码耗时合计：{a["encode_seconds_sum"]:.2f} 秒 · Python 整段解码耗时合计：{a["decode_seconds_sum"]:.2f} 秒。结果按匿名动作 ID 汇总。</text>
<text x="54" y="694" class="note">Environment: Python {escape(report["environment"]["python"])} · {escape(report["environment"]["platform"])}</text>
</svg>'''


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_frames < 2:
        raise ValueError("max-frames must be at least two")
    motions, skipped = _motion_inventory(args.data_root)
    settings = CodecSettings.preset(args.quality)
    jobs = [
        (identifier, str(path), args.max_frames, settings)
        for identifier, path, _ in motions
    ]
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_measure, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            samples.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)}", flush=True)
    samples.sort(key=lambda item: str(item["id"]))
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "codec": {"version": "SKAC v1", "quality": args.quality},
        "input": {
            "format": "numeric_smplx_npz",
            "joint_parameter_count": 55,
            "source_paths_in_report": False,
            "original_names_in_report": False,
            "non_motion_files_skipped": len(skipped),
            "skipped_ids": skipped,
            "source_frame_count": sum(item[2] for item in motions),
            "sampling": {
                "policy": "deterministic_center_window",
                "maximum_frames_per_motion": args.max_frames,
                "all_valid_motions_included": True,
            },
        },
        "metric_scope": {
            "rotation": "source_parameter_exact",
            "root_translation": "source_parameter_exact",
            "global_position": "canonical_topology_proxy_not_body_model_or_mesh_error",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "workers": args.workers,
            "wall_seconds": time.perf_counter() - started,
        },
        "aggregate": _aggregate(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    args.visual.write_text(_svg(report), encoding="utf-8", newline="\n")
    print(json.dumps({"aggregate": report["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
