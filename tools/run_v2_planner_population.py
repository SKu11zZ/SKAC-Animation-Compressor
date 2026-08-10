from __future__ import annotations

import argparse
import json
import platform
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

from skac_codec.adaptive import build_adaptive_plan  # noqa: E402
from skac_codec.format import CodecSettings  # noqa: E402
from skac_codec.model import MotionClip  # noqa: E402
from skac_codec.smpl import read_smpl_npz  # noqa: E402
from tools.run_smpl_codec_benchmark import _motion_inventory  # noqa: E402


SCHEMA = "skac.v2_planner_population"
SCHEMA_VERSION = "1.0.0"
PERCENTILES = (50, 80, 95)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the v2 planner on a deterministic public motion population."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--min-segment-frames", type=int, default=8)
    parser.add_argument("--max-segment-frames", type=int, default=32)
    return parser.parse_args(argv)


def _stratified_selection(
    motions: Sequence[tuple[str, Path, int]], count: int
) -> list[tuple[str, Path, int]]:
    if count < 1 or count > len(motions):
        raise ValueError("sample count must be between one and the valid motion count")
    ordered = sorted(motions, key=lambda item: (item[2], item[0]))
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(np.int64)
    if len(set(int(item) for item in indices)) != count:
        raise AssertionError("stratified selection produced duplicate indices")
    return [ordered[int(index)] for index in indices]


def _config(
    settings: CodecSettings,
    max_frames: int,
    min_segment_frames: int,
    max_segment_frames: int,
) -> dict[str, Any]:
    return {
        "quality": settings.quality_name,
        "max_frames": max_frames,
        "min_segment_frames": min_segment_frames,
        "max_segment_frames": max_segment_frames,
    }


def _measure(job: tuple[str, str, int, CodecSettings, int, int]) -> dict[str, Any]:
    identifier, source, max_frames, settings, min_segment_frames, max_segment_frames = job
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
    plan, _ = build_adaptive_plan(
        clip,
        settings=settings,
        min_segment_frames=min_segment_frames,
        max_segment_frames=max_segment_frames,
    )
    planner_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    planned_bytes = sum(
        int(plan["summary"][field])
        for field in (
            "planned_rotation_payload_bytes_pre_entropy",
            "planned_translation_payload_bytes_pre_entropy",
        )
    )
    raw_bytes = clip.raw_channel_bytes_float32
    return {
        "id": identifier,
        "config": _config(
            settings, max_frames, min_segment_frames, max_segment_frames
        ),
        "source_frame_count": full_clip.frame_count,
        "window_start_frame": start,
        "frame_count": clip.frame_count,
        "joint_count": clip.skeleton.joint_count,
        "duration_seconds": clip.duration_seconds,
        "passed": bool(plan["passed"]),
        "planner_ms": planner_ms,
        "planned_payload_bytes_pre_entropy": planned_bytes,
        "raw_channel_bytes_float32": raw_bytes,
        "planned_payload_percent_of_float32": planned_bytes * 100.0 / raw_bytes,
        "rotation_error_degrees_max": float(
            plan["metrics"]["rotation_error_degrees_max"]
        ),
        "global_position_error_max_height_fraction": float(
            plan["metrics"]["global_position_error_max_skeleton_height_fraction"]
        ),
        "rotation_bits_weighted_mean": float(
            plan["summary"]["planned_rotation_bits_weighted_mean"]
        ),
        "rotation_key_count": int(plan["summary"]["planned_rotation_key_count"]),
        "segment_count": int(plan["summary"]["segment_count"]),
        "rotation_budget_attempts": int(
            plan["summary"]["rotation_budget_attempts"]
        ),
        "plan_sha256": str(plan["plan_sha256"]),
    }


def _load_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"checkpoint line {line_number} is invalid JSON") from error
        if not isinstance(item, dict) or item.get("config") != config:
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"checkpoint line {line_number} has an invalid id")
        records[identifier] = item
    return records


def _append_checkpoint(path: Path, sample: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _distribution(samples: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    values = np.asarray([float(item[field]) for item in samples], dtype=np.float64)
    return {
        **{f"p{percentile}": float(np.percentile(values, percentile)) for percentile in PERCENTILES},
        "maximum": float(np.max(values)),
    }


def _aggregate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("population aggregation requires at least one sample")
    fields = (
        "planner_ms",
        "planned_payload_percent_of_float32",
        "rotation_error_degrees_max",
        "global_position_error_max_height_fraction",
        "rotation_bits_weighted_mean",
        "rotation_budget_attempts",
    )
    return {
        "sample_count": len(samples),
        "evaluated_frame_count": sum(int(item["frame_count"]) for item in samples),
        "source_frame_count": sum(int(item["source_frame_count"]) for item in samples),
        "quality_pass_count": sum(bool(item["passed"]) for item in samples),
        "quality_pass_coverage_percent": 100.0
        * sum(bool(item["passed"]) for item in samples)
        / len(samples),
        "distributions": {field: _distribution(samples, field) for field in fields},
    }


def _representative_samples(
    samples: Sequence[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    ordered = sorted(samples, key=lambda item: (float(item[field]), str(item["id"])))
    values = np.asarray([float(item[field]) for item in ordered], dtype=np.float64)
    targets = (
        ("best", float(values[0])),
        ("p50", float(np.percentile(values, 50))),
        ("p80", float(np.percentile(values, 80))),
        ("p95", float(np.percentile(values, 95))),
        ("worst", float(values[-1])),
    )
    available = list(ordered)
    result: list[dict[str, Any]] = []
    for label, target in targets:
        pool = available or list(ordered)
        selected = min(
            pool,
            key=lambda item: (
                abs(float(item[field]) - target),
                str(item["id"]),
            ),
        )
        if selected in available:
            available.remove(selected)
        result.append(
            {
                "label": label,
                "id": str(selected["id"]),
                "value": float(selected[field]),
                "frame_count": int(selected["frame_count"]),
            }
        )
    return result


def _svg(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    distributions = aggregate["distributions"]
    rows = (
        ("Planner time / 规划耗时", "planner_ms", "ms", 1.0),
        ("Pre-entropy payload / 熵编码前载荷", "planned_payload_percent_of_float32", "%", 1.0),
        ("Rotation error / 旋转误差", "rotation_error_degrees_max", "°", 1.0),
        ("Global error / 全局误差", "global_position_error_max_height_fraction", "height", 1.0),
    )
    row_svg: list[str] = []
    for index, (label, field, unit, scale) in enumerate(rows):
        y = 330 + index * 68
        values = distributions[field]
        row_svg.append(
            f'<text x="54" y="{y}" class="row">{escape(label)}</text>'
            + "".join(
                f'<text x="{x}" y="{y}" class="number">{values[key] * scale:.4f} {unit}</text>'
                for x, key in zip((550, 760, 970, 1160), ("p50", "p80", "p95", "maximum"), strict=True)
            )
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<style>.bg{{fill:#0e1012}}.accent{{fill:#25e6c8}}.title{{fill:#f4f5f6;font:700 34px Arial,sans-serif}}.sub{{fill:#989fa8;font:15px Arial,sans-serif}}.card{{fill:#171a1e;stroke:#30343a}}.value{{fill:#f4f5f6;font:700 30px Arial,sans-serif}}.label{{fill:#9fa6af;font:700 12px Arial,sans-serif}}.head{{fill:#25e6c8;font:700 12px Arial,sans-serif}}.row{{fill:#d8dbe0;font:14px Arial,sans-serif}}.number{{fill:#f4f5f6;font:13px Arial,sans-serif;text-anchor:end}}.note{{fill:#858c96;font:12px Arial,sans-serif}}</style>
<rect width="1280" height="720" class="bg"/><rect width="12" height="720" class="accent"/>
<text x="54" y="64" class="title">SKAC v2.1 Population Calibration / 人口分布校准</text>
<text x="54" y="98" class="sub">Deterministic duration-stratified public parameter streams · center window · high quality</text>
<rect x="54" y="140" width="360" height="110" rx="14" class="card"/><text x="78" y="174" class="label">SAMPLES / 样本</text><text x="78" y="222" class="value">{aggregate["sample_count"]}</text>
<rect x="438" y="140" width="360" height="110" rx="14" class="card"/><text x="462" y="174" class="label">QUALITY COVERAGE / 质量覆盖率</text><text x="462" y="222" class="value">{aggregate["quality_pass_coverage_percent"]:.1f}%</text>
<rect x="822" y="140" width="380" height="110" rx="14" class="card"/><text x="846" y="174" class="label">EVALUATED FRAMES / 评测帧数</text><text x="846" y="222" class="value">{aggregate["evaluated_frame_count"]:,}</text>
<text x="550" y="286" class="head" text-anchor="end">P50</text><text x="760" y="286" class="head" text-anchor="end">P80</text><text x="970" y="286" class="head" text-anchor="end">P95</text><text x="1160" y="286" class="head" text-anchor="end">MAX / 最差</text>
{''.join(row_svg)}
<text x="54" y="650" class="note">Payload percentage is a planner estimate before entropy coding and container overhead; it is not the final .skac file size.</text>
<text x="54" y="676" class="note">载荷占比为熵编码和容器开销之前的规划估算，不代表最终 .skac 文件体积。每条结果使用中性 ID，报告不包含源路径。</text>
</svg>'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.max_frames < 2:
        raise ValueError("workers must be positive and max-frames must be at least two")
    motions, skipped = _motion_inventory(args.data_root)
    selected = _stratified_selection(motions, args.sample_count)
    settings = CodecSettings.preset(args.quality)
    config = _config(
        settings,
        args.max_frames,
        args.min_segment_frames,
        args.max_segment_frames,
    )
    completed = _load_checkpoint(args.checkpoint, config)
    selected_ids = {item[0] for item in selected}
    samples = [completed[item] for item in sorted(selected_ids & completed.keys())]
    jobs = [
        (
            identifier,
            str(path),
            args.max_frames,
            settings,
            args.min_segment_frames,
            args.max_segment_frames,
        )
        for identifier, path, _ in selected
        if identifier not in completed
    ]
    previous_elapsed: float | None = None
    if args.output.is_file():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
            measurement = previous.get("measurement", {})
            value = measurement.get(
                "population_compute_elapsed_seconds",
                measurement.get("elapsed_seconds"),
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                previous_elapsed = float(value)
        except (json.JSONDecodeError, OSError, AttributeError):
            previous_elapsed = None
    started = time.perf_counter()
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {executor.submit(_measure, job): job[0] for job in jobs}
            for index, future in enumerate(as_completed(pending), 1):
                sample = future.result()
                _append_checkpoint(args.checkpoint, sample)
                samples.append(sample)
                print(
                    f"completed {len(samples)}/{len(selected)} ({pending[future]})",
                    file=sys.stderr,
                    flush=True,
                )
    samples.sort(key=lambda item: str(item["id"]))
    if {item["id"] for item in samples} != selected_ids:
        raise AssertionError("population run did not return the selected sample set")
    aggregate = _aggregate(samples)
    elapsed = time.perf_counter() - started
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "passed": aggregate["quality_pass_count"] == len(samples),
        "codec": {"version": "SKAC v2.1", "quality": settings.quality_name},
        "input": {
            "format": "numeric_smplx_npz",
            "valid_motion_count": len(motions),
            "non_motion_files_skipped": len(skipped),
            "source_paths_in_report": False,
            "original_names_in_report": False,
            "sampling": {
                "policy": "deterministic_duration_stratified_even_indices",
                "sample_count": len(selected),
                "maximum_frames_per_motion": args.max_frames,
                "window": "center",
                "selected_ids": [item[0] for item in selected],
            },
        },
        "measurement": {
            "population_workers": args.workers,
            "planner_time_scope": "plan_plus_embedded_v1_size_reference",
            "planner_timing": (
                "per_clip_wall_time_under_population_parallel_load_"
                "not_isolated_runtime_latency"
            ),
            "population_compute_elapsed_seconds": (
                previous_elapsed
                if not jobs and previous_elapsed is not None
                else elapsed
            ),
            "current_invocation_computed_count": len(jobs),
            "checkpoint_records_used": len(selected_ids & completed.keys()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "aggregate": aggregate,
        "representative_samples": {
            "selection_field": "planned_payload_percent_of_float32",
            "items": _representative_samples(
                samples, "planned_payload_percent_of_float32"
            ),
        },
        "samples": samples,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.visual.write_text(_svg(report), encoding="utf-8", newline="\n")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
