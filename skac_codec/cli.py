from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bvh import read_bvh, write_bvh
from .format import CodecSettings, decode_bytes, encode_bytes, inspect_file, read_skac
from .metrics import compression_metrics, roundtrip_metrics


def _settings(args: argparse.Namespace) -> CodecSettings:
    preset = CodecSettings.preset(args.quality)
    return CodecSettings(
        rotation_bits=(
            args.rotation_bits if args.rotation_bits is not None else preset.rotation_bits
        ),
        translation_bits=(
            args.translation_bits
            if args.translation_bits is not None
            else preset.translation_bits
        ),
        rotation_error_degrees=(
            args.rotation_error_degrees
            if args.rotation_error_degrees is not None
            else preset.rotation_error_degrees
        ),
        translation_error_fraction=(
            args.translation_error_fraction
            if args.translation_error_fraction is not None
            else preset.translation_error_fraction
        ),
        zlib_level=args.zlib_level if args.zlib_level is not None else preset.zlib_level,
        quality_name=args.quality,
    )


def _write_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _encode(args: argparse.Namespace) -> int:
    clip = read_bvh(args.input)
    settings = _settings(args)
    encoded = encode_bytes(clip, settings)
    decoded = decode_bytes(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    _write_json(
        {
            "command": "encode",
            "quality": settings.quality_name,
            "rotation_bits": settings.rotation_bits,
            "translation_bits": settings.translation_bits,
            "rotation_error_degrees": settings.rotation_error_degrees,
            "translation_error_fraction": settings.translation_error_fraction,
            "frame_count": clip.frame_count,
            "joint_count": clip.skeleton.joint_count,
            "skeleton_sha256": clip.skeleton.signature(),
            **compression_metrics(clip, len(encoded)),
            **roundtrip_metrics(clip, decoded),
        }
    )
    return 0


def _decode(args: argparse.Namespace) -> int:
    clip = read_skac(args.input)
    write_bvh(args.output, clip)
    _write_json(
        {
            "command": "decode",
            "frame_count": clip.frame_count,
            "joint_count": clip.skeleton.joint_count,
            "frame_time": clip.frame_time,
            "skeleton_sha256": clip.skeleton.signature(),
        }
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    _write_json(inspect_file(args.input))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skac", description="Encode, decode, and inspect SKAC animation files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="encode a BVH animation")
    encode_parser.add_argument("input", type=Path)
    encode_parser.add_argument("--output", "-o", type=Path, required=True)
    encode_parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    encode_parser.add_argument("--rotation-bits", type=int)
    encode_parser.add_argument("--translation-bits", type=int)
    encode_parser.add_argument("--rotation-error-degrees", type=float)
    encode_parser.add_argument("--translation-error-fraction", type=float)
    encode_parser.add_argument("--zlib-level", type=int)
    encode_parser.set_defaults(handler=_encode)

    decode_parser = subparsers.add_parser("decode", help="decode to the source BVH skeleton")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--output", "-o", type=Path, required=True)
    decode_parser.set_defaults(handler=_decode)

    inspect_parser = subparsers.add_parser("inspect", help="inspect container metadata")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.set_defaults(handler=_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
