from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skac_codec.format import PREFIX, CodecSettings, decode_bytes, encode_bytes
from skac_codec.format_v2 import _encode_v2_0_bytes, encode_v2_bytes, inspect_v2_bytes
from skac_codec.metrics import roundtrip_metrics
from skac_codec.model import MotionClip


def _container_sizes(data: bytes) -> dict[str, int]:
    fields = PREFIX.unpack_from(data)
    return {
        "file_bytes": len(data),
        "metadata_bytes": int(fields[4]),
        "payload_bytes": int(fields[5]),
        "raw_payload_bytes": int(fields[6]),
    }


def build_compact_report(clip: MotionClip) -> dict[str, Any]:
    settings = CodecSettings.preset("high")
    v1 = encode_bytes(clip, settings)
    v2_0 = _encode_v2_0_bytes(
        clip, settings, min_segment_frames=8, max_segment_frames=24
    )
    v2_1 = encode_v2_bytes(
        clip, settings, min_segment_frames=8, max_segment_frames=24
    )
    metrics = roundtrip_metrics(clip, decode_bytes(v2_1))
    v2_1_info = inspect_v2_bytes(v2_1)
    saved = len(v2_0) - len(v2_1)
    return {
        "schema": "skac.v2_compact_optimization",
        "schema_version": "1.0.0",
        "fixture": {
            "kind": "public_synthetic",
            "frames": clip.frame_count,
            "joints": clip.skeleton.joint_count,
            "quality": "high",
            "min_segment_frames": 8,
            "max_segment_frames": 24,
        },
        "containers": [
            {"label": "SKAC v1", **_container_sizes(v1)},
            {"label": "SKAC v2.0", **_container_sizes(v2_0)},
            {
                "label": "SKAC v2.1",
                **_container_sizes(v2_1),
                "chunk_count": int(v2_1_info["codec"]["chunk_count"]),
                "segment_count": int(v2_1_info["codec"]["segment_count"]),
            },
        ],
        "optimization": {
            "bytes_saved_vs_v2_0": saved,
            "percent_smaller_vs_v2_0": float(saved * 100.0 / len(v2_0)),
        },
        "reconstruction": metrics,
        "scope": (
            "Deterministic public synthetic format fixture; size and reconstruction "
            "evidence only, not a production corpus benchmark."
        ),
    }


def write_compact_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_compact_report_svg(path: Path, report: dict[str, Any]) -> None:
    rows = report["containers"]
    maximum = max(int(item["file_bytes"]) for item in rows)
    accent = "#ff5c35"
    bar_x = 252
    bar_width = 700
    row_markup: list[str] = []
    for index, item in enumerate(rows):
        y = 262 + index * 92
        width = int(round(int(item["file_bytes"]) * bar_width / maximum))
        color = accent if item["label"] == "SKAC v2.1" else "#3a3a3a"
        row_markup.append(
            f'<text x="54" y="{y + 24}" class="label">{item["label"]}</text>'
            f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="38" rx="4" fill="#202020"/>'
            f'<rect x="{bar_x}" y="{y}" width="{width}" height="38" rx="4" fill="{color}"/>'
            f'<text x="{bar_x + width + 16}" y="{y + 26}" class="value">{int(item["file_bytes"]):,} B</text>'
            f'<text x="{bar_x}" y="{y + 62}" class="detail">metadata {int(item["metadata_bytes"]):,} B  ·  payload {int(item["payload_bytes"]):,} B</text>'
        )
    optimization = report["optimization"]
    reconstruction = report["reconstruction"]
    fixture = report["fixture"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650">
<style>
  .title {{ fill:#f4f1eb; font:700 34px Arial,"Microsoft YaHei",sans-serif; }}
  .sub {{ fill:#a9a6a0; font:16px Arial,"Microsoft YaHei",sans-serif; }}
  .label {{ fill:#f4f1eb; font:700 19px Arial,"Microsoft YaHei",sans-serif; }}
  .value {{ fill:#f4f1eb; font:700 18px Consolas,monospace; }}
  .detail {{ fill:#92908b; font:14px Consolas,"Microsoft YaHei",monospace; }}
  .metric {{ fill:#f4f1eb; font:700 25px Consolas,monospace; }}
  .metric-label {{ fill:#a9a6a0; font:13px Arial,"Microsoft YaHei",sans-serif; }}
</style>
<rect width="1200" height="650" fill="#111111"/>
<rect x="0" y="0" width="10" height="650" fill="{accent}"/>
<text x="54" y="66" class="title">SKAC v2.1 Compact Container / 紧凑容器优化</text>
<text x="54" y="101" class="sub">Concrete file bytes · same public synthetic clip · identical high-quality gate</text>
<text x="54" y="128" class="sub">具体文件大小 · 同一公开合成动画 · 相同 high 质量门槛</text>
<rect x="54" y="158" width="1092" height="72" rx="8" fill="#181818" stroke="#303030"/>
<text x="82" y="193" class="metric">{int(optimization["bytes_saved_vs_v2_0"]):,} B</text>
<text x="82" y="216" class="metric-label">saved vs v2.0 / 相比 v2.0 节省</text>
<text x="354" y="193" class="metric">{float(optimization["percent_smaller_vs_v2_0"]):.1f}%</text>
<text x="354" y="216" class="metric-label">smaller vs v2.0 / 文件缩小</text>
<text x="626" y="193" class="metric">{float(reconstruction["rotation_error_degrees_max"]):.5f}°</text>
<text x="626" y="216" class="metric-label">max rotation error / 最大旋转误差</text>
<text x="928" y="193" class="metric">{float(reconstruction["global_position_error_max"]):.5f}</text>
<text x="928" y="216" class="metric-label">max global position error / 最大全局位置误差</text>
{''.join(row_markup)}
<line x1="54" y1="559" x2="1146" y2="559" stroke="#303030"/>
<text x="54" y="592" class="sub">Fixture: {fixture["frames"]} frames · {fixture["joints"]} joints · {rows[2]["segment_count"]} segments · {rows[2]["chunk_count"]} chunks</text>
<text x="54" y="620" class="sub">Public synthetic regression fixture; this is format evidence, not a production corpus benchmark.</text>
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
