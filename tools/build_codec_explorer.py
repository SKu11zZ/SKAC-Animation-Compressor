from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "skac.codec_explorer"
SCHEMA_VERSION = "1.0.0"
BEGIN_MARKER = "<!-- BEGIN CODEC EXPLORER DATA -->"
END_MARKER = "<!-- END CODEC EXPLORER DATA -->"
SAMPLE_FIELDS = (
    "animation",
    "character",
    "frame_count",
    "frame_time",
    "playback_budget_seconds",
    "joint_count",
    "raw_channel_bytes_float32",
    "encoded_bytes",
    "bits_per_joint_per_frame",
    "encode_clip_ms",
    "decode_clip_ms_median",
    "rotation_error_degrees_mean",
    "rotation_error_degrees_max",
    "root_translation_error_mean",
    "root_translation_error_max",
    "root_translation_error_max_height_fraction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the public, browser-safe data bundle for the Codec Explorer."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/codec_showcase_8x20_public.json"),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("reports/codec_explorer.html"),
    )
    return parser.parse_args(argv)


def build_explorer_data(report: dict[str, Any], source_bytes: bytes) -> dict[str, Any]:
    samples = report.get("samples")
    sampling = report.get("sampling")
    overall = report.get("overall")
    if not isinstance(samples, list) or not samples:
        raise ValueError("showcase report contains no samples")
    if not isinstance(sampling, dict) or not isinstance(overall, dict):
        raise ValueError("showcase report is missing aggregate metadata")

    compact_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} is not an object")
        missing = [field for field in SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"sample {index} is missing fields: {', '.join(missing)}")
        compact_samples.append({field: sample[field] for field in SAMPLE_FIELDS})

    animations = sorted({str(sample["animation"]) for sample in compact_samples})
    characters = list(dict.fromkeys(str(sample["character"]) for sample in compact_samples))
    expected = len(animations) * len(characters)
    if len(compact_samples) != expected:
        raise ValueError(
            f"expected a complete animation/character matrix of {expected} samples; "
            f"found {len(compact_samples)}"
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": int(sampling["seed"]),
        "quality": str(report["codec"]["quality"]),
        "decode_iterations": int(report["decode_iterations"]),
        "animations": animations,
        "characters": characters,
        "overall": {
            "sample_count": int(overall["sample_count"]),
            "playback_budget_seconds": float(overall["playback_budget_seconds"]),
            "raw_channel_bytes_float32": int(overall["raw_channel_bytes_float32"]),
            "encoded_bytes": int(overall["encoded_bytes"]),
            "rotation_error_degrees_max": float(
                overall["rotation_error_degrees_max"]
            ),
        },
        "samples": compact_samples,
    }


def write_html_bundle(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    html = path.read_text(encoding="utf-8")
    if html.count(BEGIN_MARKER) != 1 or html.count(END_MARKER) != 1:
        raise ValueError("Codec Explorer HTML has invalid data markers")
    before, remainder = html.split(BEGIN_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    embedded = (
        f"{BEGIN_MARKER}\n"
        f'<script id="codecExplorerData" type="application/json">{payload}</script>\n'
        f"{END_MARKER}"
    )
    path.write_text(
        before + embedded + after,
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_bytes = args.input.read_bytes()
    report = json.loads(source_bytes.decode("utf-8"))
    data = build_explorer_data(report, source_bytes)
    write_html_bundle(args.html, data)
    print(
        json.dumps(
            {
                "animation_count": len(data["animations"]),
                "character_count": len(data["characters"]),
                "sample_count": len(data["samples"]),
                "output": str(args.html),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
