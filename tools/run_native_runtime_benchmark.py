from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

import numpy as np

from skac_codec.format import CodecSettings, _read_container, encode_bytes
from skac_codec.model import MotionClip, Skeleton
from skac_codec.retarget import build_retarget_profile
from skac_codec.runtime import runtime_skeleton_bytes


REPORT_SCHEMA = "skac.native_runtime_benchmark"
REPORT_VERSION = "1.0.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark same- and different-character native runtime sampling."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--max-same-p95-ms", type=float, default=0.10)
    parser.add_argument("--max-different-p95-ms", type=float, default=0.25)
    parser.add_argument("--max-100-tick-p95-ms", type=float, default=16.667)
    return parser.parse_args(argv)


def generated_fixture() -> tuple[MotionClip, Skeleton]:
    source_joint_count = 65
    frame_count = 300
    names = tuple(["Root"] + [f"Joint{index:03d}" for index in range(1, source_joint_count)])
    parents = np.arange(-1, source_joint_count - 1, dtype=np.int32)
    offsets = np.zeros((source_joint_count, 3), dtype=np.float64)
    offsets[1:, 1] = 1.0
    rotation_channels = ("Zrotation", "Xrotation", "Yrotation")
    channels = (("Xposition", "Yposition", "Zposition") + rotation_channels,) + (
        (rotation_channels,) * (source_joint_count - 1)
    )
    skeleton = Skeleton(
        names=names,
        parents=parents,
        offsets=offsets,
        channels=channels,
        end_site_offsets=np.zeros((source_joint_count, 3), dtype=np.float64),
        has_end_sites=np.zeros(source_joint_count, dtype=np.bool_),
    )
    phase = np.linspace(0.0, 8.0 * math.pi, frame_count, dtype=np.float64)
    rotations = np.zeros((frame_count, source_joint_count, 4), dtype=np.float64)
    for joint in range(source_joint_count):
        angle = 0.18 * np.sin(phase + joint * 0.13)
        rotations[:, joint, 0] = np.cos(angle * 0.5)
        rotations[:, joint, 1 + joint % 3] = np.sin(angle * 0.5)
    translations = np.broadcast_to(offsets, (frame_count, source_joint_count, 3)).copy()
    translations[:, 0, 0] = np.linspace(0.0, 12.0, frame_count)
    clip = MotionClip(skeleton, rotations, translations, 1.0 / 30.0)

    target_names = names + ("AccessoryA", "AccessoryB")
    target_parents = np.concatenate((parents, np.asarray([0, 0], dtype=np.int32)))
    target_offsets = np.concatenate(
        (offsets * 1.15, np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=np.float64)), axis=0
    )
    target = Skeleton(
        names=target_names,
        parents=target_parents,
        offsets=target_offsets,
        channels=channels + (rotation_channels, rotation_channels),
        end_site_offsets=np.zeros((len(target_names), 3), dtype=np.float64),
        has_end_sites=np.zeros(len(target_names), dtype=np.bool_),
    )
    return clip, target


def run_benchmark(executable: Path, iterations: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source, target = generated_fixture()
    encoded = encode_bytes(source, CodecSettings.preset("high"))
    metadata, payload = _read_container(encoded)
    profile = build_retarget_profile(source.skeleton, target)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata_path = root / "metadata.json"
        payload_path = root / "payload.bin"
        profile_path = root / "profile.json"
        target_path = root / "target.json"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        payload_path.write_bytes(payload)
        profile_path.write_text(
            json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        target_path.write_bytes(runtime_skeleton_bytes(target))
        completed = subprocess.run(
            [
                str(executable.resolve()),
                str(metadata_path),
                str(payload_path),
                str(profile_path),
                str(target_path),
                str(iterations),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    results: list[dict[str, Any]] = []
    fixture = {
        "frame_count": source.frame_count,
        "source_joint_count": source.skeleton.joint_count,
        "target_joint_count": target.joint_count,
        "mapped_joint_count": len(profile.transfers),
    }
    for line in completed.stdout.splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"same", "different"}:
            continue
        results.append(
            {
                "mode": "same_character" if fields[0] == "same" else "different_character",
                "instances": int(fields[1]),
                "sample_ms_p50": float(fields[2]),
                "sample_ms_p95": float(fields[3]),
                "sample_ms_p99": float(fields[4]),
                "tick_ms_p95": float(fields[5]),
            }
        )
    if len(results) != 5:
        raise RuntimeError("native benchmark did not return the expected five records")
    return results, fixture


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    results, fixture = run_benchmark(args.benchmark, args.iterations)
    same = next(item for item in results if item["mode"] == "same_character")
    different_one = next(
        item for item in results if item["mode"] == "different_character" and item["instances"] == 1
    )
    different_hundred = next(
        item for item in results if item["mode"] == "different_character" and item["instances"] == 100
    )
    checks = [
        {
            "id": "same_character_sample_p95",
            "value_ms": same["sample_ms_p95"],
            "maximum_ms": args.max_same_p95_ms,
            "passed": same["sample_ms_p95"] <= args.max_same_p95_ms,
        },
        {
            "id": "different_character_sample_p95",
            "value_ms": different_one["sample_ms_p95"],
            "maximum_ms": args.max_different_p95_ms,
            "passed": different_one["sample_ms_p95"] <= args.max_different_p95_ms,
        },
        {
            "id": "different_character_100_instances_tick_p95",
            "value_ms": different_hundred["tick_ms_p95"],
            "maximum_ms": args.max_100_tick_p95_ms,
            "passed": different_hundred["tick_ms_p95"] <= args.max_100_tick_p95_ms,
        },
    ]
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "fixture": fixture,
        "method": {
            "iterations": args.iterations,
            "sampling": "time_interpolated_loop",
            "same_character_warmup_ticks": 200,
            "different_character_warmup_ticks": 100,
            "file_io_included": False,
            "fixture_origin": "deterministically_generated_public_safe",
            "build_type": args.build_type,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "results": results,
        "checks": checks,
        "notes": [
            "Measurements are machine-dependent and are not cross-machine rankings.",
            "The same-character API and format-1.0 container remain unchanged.",
            "Each retarget playback instance owns reusable scratch buffers; sampling performs no explicit allocation.",
        ],
    }


def text(x: float, y: float, value: str, css_class: str, anchor: str | None = None) -> str:
    anchor_value = "" if anchor is None else f' text-anchor="{anchor}"'
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}"{anchor_value}>{escape(value)}</text>'


def write_visual(path: Path, report: dict[str, Any]) -> None:
    width, height = 1400, 780
    ink, muted, grid = "#17202a", "#667085", "#d7dee7"
    blue, blue_open, gold = "#2864dc", "#dbe8ff", "#c98a16"
    results = report["results"]
    iterations = int(report["method"]["iterations"])
    same = results[0]
    different = [item for item in results if item["mode"] == "different_character"]
    max_sample = max(item["sample_ms_p95"] for item in results) * 1.2
    max_tick = max(16.667, max(item["tick_ms_p95"] for item in different)) * 1.08
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1400" height="780" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Segoe UI,Noto Sans SC,Microsoft YaHei,Arial,sans-serif}.title{font-size:32px;font-weight:750;fill:#17202a}.sub{font-size:14px;fill:#667085}.kpi{font-size:29px;font-weight:750;fill:#17202a}.label{font-size:12px;font-weight:650;fill:#667085}.panel{font-size:18px;font-weight:700;fill:#17202a}.row{font-size:13px;font-weight:600;fill:#17202a}.value{font-size:12px;font-weight:700;fill:#17202a}.note{font-size:12px;fill:#667085}</style>',
        text(54, 54, "Native Runtime Sampling / 原生运行时采样", "title"),
        text(54, 82, "Generated 300-frame fixture · 65 source joints → 67 target joints · Release build · I/O excluded", "sub"),
        text(54, 104, "生成型 300 帧夹具 · 65 源关节 → 67 目标关节 · Release 构建 · 不含文件读写", "sub"),
    ]
    cards = [
        (f'{same["sample_ms_p95"] * 1000:.2f} μs', "SAME CHARACTER P95", "同角色单次采样 P95"),
        (f'{different[0]["sample_ms_p95"] * 1000:.2f} μs', "PROFILE PLAYBACK P95", "跨骨骼单次采样 P95"),
        (f'{different[-1]["tick_ms_p95"]:.3f} ms', "100 INSTANCES TICK P95", "100 实例整帧 P95"),
        ("PASS" if report["passed"] else "FAIL", "QUALITY GATES", "运行时质量门槛"),
    ]
    for index, (value, label, chinese) in enumerate(cards):
        x = 54 + index * 332
        lines.append(f'<rect x="{x}" y="132" width="306" height="112" rx="10" fill="#f7f9fc" stroke="{grid}"/>')
        lines.append(text(x + 20, 178, value, "kpi"))
        lines.append(text(x + 20, 207, label, "label"))
        lines.append(text(x + 20, 228, chinese, "sub"))

    lines.extend([
        text(54, 294, "Per-sample P95 / 单次采样 P95", "panel"),
        text(54, 317, "Time-interpolated sampling; lower is better / 时间插值采样，越低越好", "sub"),
        text(746, 294, "Different-character tick P95 / 跨骨骼整帧 P95", "panel"),
        text(746, 317, "Sequential playback instances; 16.667 ms frame reference / 串行播放实例；16.667 ms 帧预算参考", "sub"),
    ])
    left_x, right_x, plot_width = 208.0, 902.0, 400.0
    row_start, row_gap, bar_height = 360.0, 60.0, 24.0
    left_rows = [("Same ×1 / 同角色", same)] + [
        (f'Profile ×{item["instances"]} / 跨骨骼', item) for item in different
    ]
    for index, (label, item) in enumerate(left_rows):
        y = row_start + index * row_gap
        bar_width = plot_width * item["sample_ms_p95"] / max_sample
        lines.append(text(left_x - 14, y + 17, label, "row", "end"))
        lines.append(f'<rect x="{left_x}" y="{y}" width="{plot_width}" height="{bar_height}" rx="4" fill="{blue_open}"/>')
        lines.append(f'<rect x="{left_x}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="4" fill="{blue}"/>')
        lines.append(text(left_x + bar_width + 8, y + 17, f'{item["sample_ms_p95"] * 1000:.2f} μs', "value"))
    for index, item in enumerate(different):
        y = row_start + index * 75
        bar_width = plot_width * item["tick_ms_p95"] / max_tick
        lines.append(text(right_x - 14, y + 17, f'{item["instances"]} instances / 实例', "row", "end"))
        lines.append(f'<rect x="{right_x}" y="{y}" width="{plot_width}" height="{bar_height}" rx="4" fill="#f8e7bd"/>')
        lines.append(f'<rect x="{right_x}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="4" fill="{gold}"/>')
        lines.append(text(right_x + bar_width + 8, y + 17, f'{item["tick_ms_p95"]:.3f} ms', "value"))
    gate_x = right_x + plot_width * 16.667 / max_tick
    lines.append(f'<line x1="{gate_x:.1f}" y1="344" x2="{gate_x:.1f}" y2="648" stroke="{ink}" stroke-width="2" stroke-dasharray="5 5"/>')
    lines.append(text(gate_x - 6, 668, "16.667 ms / 60 FPS", "note", "end"))
    lines.extend([
        text(54, 724, f"P95 from {iterations:,} measured ticks after warmup. Results are machine-dependent; use them as local regression evidence, not a hardware ranking.", "note"),
        text(54, 748, f"预热后统计 {iterations:,} 个 Tick 的 P95。结果与机器有关，只用于本地回归，不用于硬件排名。", "note"),
        "</svg>",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iterations < 100:
        raise SystemExit("--iterations must be at least 100")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    write_visual(args.visual, report)
    print(json.dumps({"passed": report["passed"], "output": str(args.output), "visual": str(args.visual)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
