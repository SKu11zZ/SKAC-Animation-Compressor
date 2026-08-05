from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .bvh import read_bvh, write_bvh
from .fbx import extract_fbx_to_bvh, inject_bvh_into_fbx, validate_fbx
from .format import CodecSettings, decode_bytes, encode_bytes, inspect_file, read_skac
from .metrics import compression_metrics, roundtrip_metrics
from .pack import inspect_pack_file, read_pack, write_pack
from .quality import (
    QualityThresholds,
    run_quality_gate,
    write_quality_report_json,
    write_quality_report_svg,
)
from .retarget import (
    build_retarget_profile,
    load_retarget_profile,
    retarget_motion,
    save_retarget_profile,
)
from .runtime import save_runtime_skeleton


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


def _pack_entries(values: list[str]) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for value in values:
        entry_id, separator, supplied_path = value.partition("=")
        if not separator or not entry_id or not supplied_path:
            raise ValueError("--clip must use ID=PATH")
        if entry_id in entries:
            raise ValueError(f"duplicate pack entry id: {entry_id}")
        path = Path(supplied_path)
        if not path.is_file():
            raise ValueError(f"pack input does not exist: {path}")
        entries[entry_id] = path.read_bytes()
    return entries


def _pack_create(args: argparse.Namespace) -> int:
    entries = _pack_entries(args.clip)
    write_pack(args.output, entries)
    _write_json({"command": "pack-create", **inspect_pack_file(args.output)})
    return 0


def _pack_inspect(args: argparse.Namespace) -> int:
    _write_json({"command": "pack-inspect", **inspect_pack_file(args.input)})
    return 0


def _pack_extract(args: argparse.Namespace) -> int:
    archive = read_pack(args.input)
    encoded = archive.entry_bytes(args.entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    _write_json(
        {
            "command": "pack-extract",
            "entry": args.entry,
            "output": str(args.output),
            "bytes": len(encoded),
        }
    )
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
            "profile_schema_version": profile.schema_version,
            "profile_sha256": profile.signature(),
            "source_skeleton_sha256": profile.source_skeleton_sha256,
            "target_skeleton_sha256": profile.target_skeleton_sha256,
            "mapped_joint_count": len(profile.transfers),
            "shared_core_joint_count": profile.shared_core_joint_count,
            "mapped_core_joint_count": profile.mapped_core_joint_count,
            "core_coverage": profile.core_coverage,
            "source_runtime_joint_count": len(profile.source_evaluation_order),
            "target_runtime_joint_count": len(profile.target_evaluation_order),
            "root_translation_scale": profile.root_translation_scale,
            "contact_lock": profile.contact_lock,
            "foot_pair_count": len(profile.foot_pairs),
        }
    )
    return 0


def _runtime_skeleton(args: argparse.Namespace) -> int:
    skeleton = read_bvh(args.target).skeleton
    save_runtime_skeleton(args.output, skeleton)
    _write_json(
        {
            "command": "runtime-skeleton",
            "joint_count": skeleton.joint_count,
            "skeleton_sha256": skeleton.signature(),
        }
    )
    return 0


def _fbx_extract(args: argparse.Namespace) -> int:
    report = extract_fbx_to_bvh(
        args.input,
        args.output,
        blender_executable=args.blender,
        armature_name=args.armature,
        timeout_seconds=args.timeout,
    )
    _write_json(report)
    return 0


def _fbx_inject(args: argparse.Namespace) -> int:
    report = inject_bvh_into_fbx(
        args.template,
        args.animation,
        args.output,
        blender_executable=args.blender,
        armature_name=args.armature,
        timeout_seconds=args.timeout,
    )
    _write_json(report)
    return 0


def _fbx_validate(args: argparse.Namespace) -> int:
    _write_json(
        validate_fbx(
            args.input,
            blender_executable=args.blender,
            timeout_seconds=args.timeout,
        )
    )
    return 0


def _quality_gate(args: argparse.Namespace) -> int:
    source = read_bvh(args.source)
    evaluation_case = args.evaluation_case
    target_skeleton = (
        source.skeleton
        if evaluation_case == "same_character"
        else read_bvh(args.target).skeleton
    )
    settings = _settings(args)
    thresholds = QualityThresholds.for_settings(settings)
    thresholds = replace(
        thresholds,
        minimum_core_coverage=getattr(
            args, "minimum_core_coverage", thresholds.minimum_core_coverage
        ),
        maximum_retarget_frame_ms_p95=getattr(
            args,
            "maximum_retarget_frame_ms",
            thresholds.maximum_retarget_frame_ms_p95,
        ),
        minimum_decode_realtime_factor=args.minimum_decode_realtime_factor,
        minimum_pipeline_realtime_factor=getattr(
            args,
            "minimum_pipeline_realtime_factor",
            thresholds.minimum_pipeline_realtime_factor,
        ),
    )
    report = run_quality_gate(
        source,
        target_skeleton,
        settings=settings,
        thresholds=thresholds,
        decode_iterations=args.decode_iterations,
        pipeline_iterations=getattr(args, "pipeline_iterations", 3),
        frame_samples=getattr(args, "frame_samples", 300),
        evaluation_case=evaluation_case,
    )
    write_quality_report_json(args.output, report)
    write_quality_report_svg(args.visual, report)
    _write_json(
        {
            "command": args.command,
            "evaluation_case": report["evaluation_case"],
            "passed": report["passed"],
            "report": str(args.output),
            "visual": str(args.visual),
            "failed_checks": [
                item["id"] for item in report["checks"] if not item["passed"]
            ],
            "performance": report["performance"],
        }
    )
    return 0 if report["passed"] else 4


def _add_quality_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument(
        "--quality", choices=("low", "medium", "high"), default="high"
    )
    parser.add_argument("--rotation-bits", type=int)
    parser.add_argument("--translation-bits", type=int)
    parser.add_argument("--rotation-error-degrees", type=float)
    parser.add_argument("--translation-error-fraction", type=float)
    parser.add_argument("--zlib-level", type=int)
    parser.add_argument(
        "--minimum-decode-realtime-factor", type=float, default=10.0
    )
    parser.add_argument("--decode-iterations", type=int, default=5)


def _add_quality_different_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--minimum-core-coverage", type=float, default=1.0)
    parser.add_argument("--maximum-retarget-frame-ms", type=float, default=4.0)
    parser.add_argument(
        "--minimum-pipeline-realtime-factor", type=float, default=5.0
    )
    parser.add_argument("--pipeline-iterations", type=int, default=3)
    parser.add_argument("--frame-samples", type=int, default=300)


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

    pack_create_parser = subparsers.add_parser(
        "pack-create", help="build a deterministic SKAC Pack from encoded animations"
    )
    pack_create_parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="add one named .skac animation; repeat for multiple entries",
    )
    pack_create_parser.add_argument("--output", "-o", type=Path, required=True)
    pack_create_parser.set_defaults(handler=_pack_create)

    pack_inspect_parser = subparsers.add_parser(
        "pack-inspect", help="inspect and fully validate a SKAC Pack"
    )
    pack_inspect_parser.add_argument("input", type=Path)
    pack_inspect_parser.set_defaults(handler=_pack_inspect)

    pack_extract_parser = subparsers.add_parser(
        "pack-extract", help="extract one named animation from a SKAC Pack"
    )
    pack_extract_parser.add_argument("input", type=Path)
    pack_extract_parser.add_argument("entry")
    pack_extract_parser.add_argument("--output", "-o", type=Path, required=True)
    pack_extract_parser.set_defaults(handler=_pack_extract)

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
        help="legacy option; rejected by real-time Profile 2.0",
    )
    profile_parser.set_defaults(handler=_profile)

    runtime_skeleton_parser = subparsers.add_parser(
        "runtime-skeleton", help="export a target skeleton for the native runtime"
    )
    runtime_skeleton_parser.add_argument("target", type=Path, help="target BVH template")
    runtime_skeleton_parser.add_argument("--output", "-o", type=Path, required=True)
    runtime_skeleton_parser.set_defaults(handler=_runtime_skeleton)

    extract_parser = subparsers.add_parser(
        "fbx-extract", help="extract an FBX armature animation to BVH through Blender"
    )
    extract_parser.add_argument("input", type=Path)
    extract_parser.add_argument("--output", "-o", type=Path, required=True)
    extract_parser.add_argument("--blender", type=Path)
    extract_parser.add_argument("--armature")
    extract_parser.add_argument("--timeout", type=int, default=300)
    extract_parser.set_defaults(handler=_fbx_extract)

    inject_parser = subparsers.add_parser(
        "fbx-inject", help="bake a target BVH animation into an FBX character"
    )
    inject_parser.add_argument("template", type=Path)
    inject_parser.add_argument("animation", type=Path)
    inject_parser.add_argument("--output", "-o", type=Path, required=True)
    inject_parser.add_argument("--blender", type=Path)
    inject_parser.add_argument("--armature")
    inject_parser.add_argument("--timeout", type=int, default=600)
    inject_parser.set_defaults(handler=_fbx_inject)

    validate_parser = subparsers.add_parser(
        "fbx-validate", help="open and inspect an FBX through Blender"
    )
    validate_parser.add_argument("input", type=Path)
    validate_parser.add_argument("--blender", type=Path)
    validate_parser.add_argument("--timeout", type=int, default=300)
    validate_parser.set_defaults(handler=_fbx_validate)

    same_quality_parser = subparsers.add_parser(
        "quality-gate-same",
        help="gate direct Codec decode back to the source character",
    )
    same_quality_parser.add_argument(
        "source", type=Path, help="public source BVH animation"
    )
    _add_quality_common_options(same_quality_parser)
    same_quality_parser.set_defaults(
        handler=_quality_gate, evaluation_case="same_character"
    )

    different_quality_parser = subparsers.add_parser(
        "quality-gate-different",
        help="gate Codec decode plus compiled playback on a different character",
    )
    different_quality_parser.add_argument(
        "source", type=Path, help="public source BVH animation"
    )
    different_quality_parser.add_argument(
        "target", type=Path, help="public target BVH template"
    )
    _add_quality_common_options(different_quality_parser)
    _add_quality_different_options(different_quality_parser)
    different_quality_parser.set_defaults(
        handler=_quality_gate, evaluation_case="different_character"
    )

    quality_parser = subparsers.add_parser(
        "quality-gate",
        help="compatibility route: classify the case from skeleton signatures",
    )
    quality_parser.add_argument("source", type=Path, help="public source BVH animation")
    quality_parser.add_argument("target", type=Path, help="public target BVH template")
    _add_quality_common_options(quality_parser)
    _add_quality_different_options(quality_parser)
    quality_parser.set_defaults(handler=_quality_gate, evaluation_case="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
