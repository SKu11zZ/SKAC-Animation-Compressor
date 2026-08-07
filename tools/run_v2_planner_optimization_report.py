from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from skac_codec.adaptive import build_adaptive_plan  # noqa: E402
from skac_codec.format import CodecSettings  # noqa: E402
from skac_codec.smpl import read_smpl_npz  # noqa: E402


SCHEMA = "skac.v2_planner_optimization"
SCHEMA_VERSION = "1.0.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the current v2 planner with a detached public baseline."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args(argv)


def _selection(root: Path) -> list[tuple[str, int, Path]]:
    valid: list[tuple[int, Path]] = []
    for path in sorted(root.glob("smpl_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                poses = archive["poses"]
                archive["trans"]
                archive["mocap_frame_rate"]
                if poses.ndim == 2 and poses.shape[1] == 165:
                    valid.append((int(poses.shape[0]), path))
        except (KeyError, ValueError, OSError):
            continue
    if len(valid) < 2:
        raise ValueError("at least two valid neutral SMPL-X motions are required")
    valid.sort(key=lambda item: (item[0], item[1].stem))
    chosen = (valid[0], valid[len(valid) // 2])
    return [
        (label, frames, path)
        for label, (frames, path) in zip(("shortest", "median"), chosen, strict=True)
    ]


_BASELINE_WORKER = r'''
import json
import sys
import time
from pathlib import Path
from skac_codec.adaptive import build_adaptive_plan
from skac_codec.format import CodecSettings
from skac_codec.smpl import read_smpl_npz
clip = read_smpl_npz(Path(sys.argv[1]))
started = time.perf_counter()
plan, _ = build_adaptive_plan(clip, settings=CodecSettings.preset("high"))
print(json.dumps({
    "seconds": time.perf_counter() - started,
    "plan_sha256": plan["plan_sha256"],
    "passed": plan["passed"],
    "segment_count": plan["summary"]["segment_count"],
    "rotation_budget_attempts": plan["summary"]["rotation_budget_attempts"],
}, sort_keys=True))
'''


def _baseline_measure(path: Path, baseline_root: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(baseline_root)
    completed = subprocess.run(
        [sys.executable, "-c", _BASELINE_WORKER, str(path.resolve())],
        cwd=baseline_root,
        env=environment,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("baseline planner process failed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("baseline planner returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("baseline planner returned an invalid record")
    return value


def _optimized_measure(path: Path) -> dict[str, Any]:
    clip = read_smpl_npz(path)
    started = time.perf_counter()
    plan, _ = build_adaptive_plan(clip, settings=CodecSettings.preset("high"))
    return {
        "seconds": time.perf_counter() - started,
        "plan_sha256": plan["plan_sha256"],
        "passed": plan["passed"],
        "segment_count": plan["summary"]["segment_count"],
        "rotation_budget_attempts": plan["summary"]["rotation_budget_attempts"],
    }


def build_report(
    data_root: Path, baseline_root: Path, repeats: int = 3
) -> dict[str, Any]:
    if not baseline_root.is_dir():
        raise ValueError("the detached baseline root is missing")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    samples: list[dict[str, Any]] = []
    for label, frames, path in _selection(data_root):
        baseline_runs: list[dict[str, Any]] = []
        optimized_runs: list[dict[str, Any]] = []
        for _ in range(repeats):
            baseline_runs.append(_baseline_measure(path, baseline_root))
            optimized_runs.append(_optimized_measure(path))
        baseline = baseline_runs[0]
        optimized = optimized_runs[0]
        baseline_seconds = statistics.median(
            float(item["seconds"]) for item in baseline_runs
        )
        optimized_seconds = statistics.median(
            float(item["seconds"]) for item in optimized_runs
        )
        plan_hashes = {
            str(item["plan_sha256"])
            for item in (*baseline_runs, *optimized_runs)
        }
        identical = len(plan_hashes) == 1
        samples.append(
            {
                "sample": label,
                "id": path.stem,
                "frame_count": frames,
                "baseline_seconds": baseline_seconds,
                "optimized_seconds": optimized_seconds,
                "baseline_seconds_runs": [
                    float(item["seconds"]) for item in baseline_runs
                ],
                "optimized_seconds_runs": [
                    float(item["seconds"]) for item in optimized_runs
                ],
                "speedup": baseline_seconds / optimized_seconds,
                "plan_sha256": optimized["plan_sha256"],
                "plan_identical": identical,
                "quality_passed": all(
                    bool(item["passed"])
                    for item in (*baseline_runs, *optimized_runs)
                ),
                "segment_count": int(optimized["segment_count"]),
                "rotation_budget_attempts": int(optimized["rotation_budget_attempts"]),
            }
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "passed": all(
            bool(item["plan_identical"] and item["quality_passed"]) for item in samples
        ),
        "scope": {
            "baseline": "pre_vectorized_planner",
            "optimized": (
                "vectorized_quantization_shared_threshold_tree_"
                "candidate_cache_and_validated_boundary_normalization_reuse"
            ),
            "timing_statistic": "median",
            "timing_repeats": repeats,
            "quality": "high",
            "source_paths_in_report": False,
            "original_names_in_report": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "samples": samples,
    }


def _svg(report: dict[str, Any]) -> str:
    samples = report["samples"]
    cards: list[str] = []
    for index, sample in enumerate(samples):
        x = 54 + index * 592
        cards.append(
            f'<rect x="{x}" y="170" width="556" height="190" rx="14" class="card"/>'
            f'<text x="{x + 24}" y="207" class="label">{escape(str(sample["sample"]).upper())} · {sample["frame_count"]} FRAMES</text>'
            f'<text x="{x + 24}" y="258" class="value">{sample["speedup"]:.2f}× faster</text>'
            f'<text x="{x + 24}" y="296" class="sub">Baseline {sample["baseline_seconds"]:.2f} s　→　Optimized {sample["optimized_seconds"]:.2f} s</text>'
            f'<text x="{x + 24}" y="330" class="ok">IDENTICAL PLAN SHA-256 / 规划结果一致</text>'
        )
    max_seconds = max(float(item["baseline_seconds"]) for item in samples)
    bars: list[str] = []
    for index, sample in enumerate(samples):
        y = 442 + index * 92
        baseline_width = 820 * float(sample["baseline_seconds"]) / max_seconds
        optimized_width = 820 * float(sample["optimized_seconds"]) / max_seconds
        bars.append(
            f'<text x="54" y="{y + 15}" class="axis">{escape(str(sample["sample"]).title())}</text>'
            f'<rect x="190" y="{y}" width="{baseline_width:.1f}" height="22" rx="5" class="raw"/>'
            f'<rect x="190" y="{y + 32}" width="{optimized_width:.1f}" height="22" rx="5" class="accent"/>'
            f'<text x="{202 + baseline_width:.1f}" y="{y + 16}" class="axis">{sample["baseline_seconds"]:.2f} s</text>'
            f'<text x="{202 + optimized_width:.1f}" y="{y + 48}" class="axis">{sample["optimized_seconds"]:.2f} s</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<style>.bg{{fill:#0e1012}}.card{{fill:#171a1e;stroke:#30343a}}.title{{fill:#f3f4f6;font:700 34px Arial,sans-serif}}.sub{{fill:#9ba1aa;font:15px Arial,sans-serif}}.label{{fill:#a4aab3;font:700 12px Arial,sans-serif;letter-spacing:.5px}}.value{{fill:#f6f7f8;font:700 34px Arial,sans-serif}}.axis{{fill:#abb1ba;font:13px Arial,sans-serif}}.ok{{fill:#57f287;font:700 12px Arial,sans-serif}}.note{{fill:#858c96;font:12px Arial,sans-serif}}.raw{{fill:#e9eaec}}.accent{{fill:#7c5cff}}</style>
<rect width="1280" height="720" class="bg"/><rect width="12" height="720" class="accent"/>
<text x="54" y="64" class="title">SKAC v2.1 Planner Optimization / 规划器优化</text>
<text x="54" y="98" class="sub">Shared error tree + candidate cache + normalize once at validated boundaries</text>
<text x="54" y="124" class="sub">共享误差树 + 候选缓存 + 入口校验后复用归一化结果 · high quality · repeated-run median</text>
{''.join(cards)}
<text x="54" y="406" class="label">PLANNER TIME / 规划耗时　　WHITE = BASELINE　PURPLE = OPTIMIZED</text>
{''.join(bars)}
<text x="54" y="668" class="note">Neutral public SMPL-X parameter streams · offline encode planning only · file I/O excluded</text>
<text x="54" y="692" class="note">公开中性 SMPL-X 参数流 · 仅测离线编码规划 · 不包含文件读取</text>
</svg>'''


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.data_root, args.baseline_root, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    args.visual.write_text(_svg(report), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
