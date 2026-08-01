from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bvh import read_bvh, write_bvh
from .format import CodecSettings, decode_bytes, encode_bytes, inspect_file, read_skac
from .metrics import compression_metrics, roundtrip_metrics
from .retarget import (
    build_retarget_profile,
    load_retarget_profile,
    retarget_motion,
    save_retarget_profile,
)


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
    diagnostics = {}
    command = "decode"
    if (args.target is None) != (args.profile is None):
        raise ValueError("--target and --profile must be supplied together")
    if args.target is not None:
        target_skeleton = read_bvh(args.target).skeleton
        profile = load_retarget_profile(args.profile)
        clip, diagnostics = retarget_motion(clip, target_skeleton, profile)
        command = "decode-retarget"
    write_bvh(args.output, clip)
    _write_json(
        {
            "command": command,
            "frame_count": clip.frame_count,
            "joint_count": clip.skeleton.joint_count,
            "frame_time": clip.frame_time,
            "skeleton_sha256": clip.skeleton.signature(),
            **diagnostics,
        }
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    _write_json(inspect_file(args.input))
    return 0


def _profile(args: argparse.Namespace) -> int:
    source = read_skac(args.source).skeleton
    target = read_bvh(args.target).skeleton
    up_axis = {"X": 0, "Y": 1, "Z": 2}[args.up_axis]
    profile = build_retarget_profile(
        source,
        target,
        up_axis=up_axis,
        contact_lock=args.contact_lock,
    )
    save_retarget_profile(args.output, profile)
    _write_json(
        {
            "command": "profile",
            "profile_sha256": profile.signature(),
            "source_skeleton_sha256": profile.source_skeleton_sha256,
            "target_skeleton_sha256": profile.target_skeleton_sha256,
            "mapped_joint_count": len(profile.transfers),
            "root_translation_scale": profile.root_translation_scale,
            "contact_lock": profile.contact_lock,
            "foot_pair_count": len(profile.foot_pairs),
        }
    )
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

    decode_parser = subparsers.add_parser(
        "decode", help="decode to the source skeleton or a profiled target skeleton"
    )
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--output", "-o", type=Path, required=True)
    decode_parser.add_argument("--target", type=Path)
    decode_parser.add_argument("--profile", type=Path)
    decode_parser.set_defaults(handler=_decode)

    inspect_parser = subparsers.add_parser("inspect", help="inspect container metadata")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.set_defaults(handler=_inspect)

    profile_parser = subparsers.add_parser(
        "profile", help="freeze a source-to-target skeleton profile"
    )
    profile_parser.add_argument("source", type=Path, help="source .skac animation")
    profile_parser.add_argument("target", type=Path, help="target BVH template")
    profile_parser.add_argument("--output", "-o", type=Path, required=True)
    profile_parser.add_argument("--up-axis", choices=("X", "Y", "Z"), default="Y")
    profile_parser.add_argument(
        "--contact-lock",
        action="store_true",
        help="enable the experimental foot-contact root correction",
    )
    profile_parser.set_defaults(handler=_profile)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
