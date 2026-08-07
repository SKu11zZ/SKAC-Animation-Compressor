from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive import (
    build_adaptive_plan,
    write_adaptive_plan_json,
    write_adaptive_plan_svg,
)
from .bvh import read_bvh, write_bvh
from .format import CodecSettings, decode_bytes, encode_bytes, inspect_file, read_skac
from .format_v2 import encode_v2_bytes
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
from .smpl import read_smpl_npz


PROTOCOL = "skac.agent.v1"
MAX_REQUEST_BYTES = 1024 * 1024


class AgentRequestError(ValueError):
    """A request that does not satisfy the public Agent protocol."""


def capabilities() -> dict[str, Any]:
    codec_gate_optional = [
        "overwrite",
        "quality",
        "rotation_bits",
        "translation_bits",
        "rotation_error_degrees",
        "translation_error_fraction",
        "zlib_level",
        "minimum_decode_realtime_factor",
        "decode_iterations",
        "format_version",
        "min_segment_frames",
        "max_segment_frames",
    ]
    different_gate_optional = codec_gate_optional + [
        "minimum_core_coverage",
        "maximum_retarget_frame_ms",
        "minimum_pipeline_realtime_factor",
        "pipeline_iterations",
        "frame_samples",
    ]
    return {
        "protocol": PROTOCOL,
        "transports": ["single-json", "json-lines"],
        "path_policy": "relative-to-workspace",
        "operations": {
            "encode": {
                "required": ["input", "output"],
                "optional": [
                    "overwrite",
                    "quality",
                    "rotation_bits",
                    "translation_bits",
                    "rotation_error_degrees",
                    "translation_error_fraction",
                    "zlib_level",
                    "format_version",
                    "min_segment_frames",
                    "max_segment_frames",
                ],
            },
            "inspect": {"required": ["input"], "optional": []},
            "pack_create": {
                "required": ["entries", "output"],
                "optional": ["overwrite"],
            },
            "pack_inspect": {"required": ["input"], "optional": []},
            "pack_extract": {
                "required": ["input", "entry", "output"],
                "optional": ["overwrite"],
            },
            "adaptive_plan": {
                "required": ["source", "report", "visual"],
                "optional": [
                    "overwrite",
                    "quality",
                    "min_segment_frames",
                    "max_segment_frames",
                ],
            },
            "decode": {
                "required": ["input", "output"],
                "optional": ["overwrite", "target", "profile"],
            },
            "profile": {
                "required": ["source", "target", "output"],
                "optional": ["overwrite", "up_axis", "contact_lock"],
            },
            "runtime_skeleton": {
                "required": ["target", "output"],
                "optional": ["overwrite"],
            },
            "quality_gate_same": {
                "required": ["source", "report", "visual"],
                "optional": codec_gate_optional,
            },
            "quality_gate_different": {
                "required": ["source", "target", "report", "visual"],
                "optional": different_gate_optional,
            },
            "quality_gate": {
                "required": ["source", "target", "report", "visual"],
                "optional": different_gate_optional,
                "compatibility": True,
            },
        },
        "exit_codes": {
            "0": "all requests succeeded",
            "2": "one or more requests were invalid",
            "3": "one or more valid operations failed",
        },
    }


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AgentRequestError(f"{label} must be a JSON object with string keys")
    return value


def _strict_fields(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise AgentRequestError(f"{label} is missing: {', '.join(missing)}")
    if unknown:
        raise AgentRequestError(f"{label} has unknown fields: {', '.join(unknown)}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentRequestError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AgentRequestError(f"{label} must be a boolean")
    return value


def _number(value: Any, label: str, kind: type[int] | type[float]) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentRequestError(f"{label} must be a number")
    if kind is int and not isinstance(value, int):
        raise AgentRequestError(f"{label} must be an integer")
    return kind(value)


def _relative_path(workspace: Path, value: Any, label: str, *, input_file: bool) -> Path:
    supplied = Path(_string(value, label))
    if supplied.is_absolute():
        raise AgentRequestError(f"{label} must be relative to the workspace")
    resolved = (workspace / supplied).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise AgentRequestError(f"{label} leaves the workspace") from error
    if input_file and not resolved.is_file():
        raise AgentRequestError(f"{label} does not name an existing file")
    return resolved


def _display_path(workspace: Path, path: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _output_path(workspace: Path, arguments: Mapping[str, Any]) -> Path:
    output = _relative_path(workspace, arguments["output"], "arguments.output", input_file=False)
    overwrite = _boolean(arguments.get("overwrite", False), "arguments.overwrite")
    if output.exists() and not overwrite:
        raise AgentRequestError("arguments.output already exists; set overwrite to true")
    if output.exists() and not output.is_file():
        raise AgentRequestError("arguments.output is not a regular file")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _atomic_write(output: Path, writer: Callable[[Path], None]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _codec_settings(arguments: Mapping[str, Any]) -> CodecSettings:
    quality = _string(arguments.get("quality", "high"), "arguments.quality")
    preset = CodecSettings.preset(quality)

    def integer(name: str, fallback: int) -> int:
        return int(_number(arguments[name], f"arguments.{name}", int)) if name in arguments else fallback

    def floating(name: str, fallback: float) -> float:
        return float(_number(arguments[name], f"arguments.{name}", float)) if name in arguments else fallback

    return CodecSettings(
        rotation_bits=integer("rotation_bits", preset.rotation_bits),
        translation_bits=integer("translation_bits", preset.translation_bits),
        rotation_error_degrees=floating(
            "rotation_error_degrees", preset.rotation_error_degrees
        ),
        translation_error_fraction=floating(
            "translation_error_fraction", preset.translation_error_fraction
        ),
        zlib_level=integer("zlib_level", preset.zlib_level),
        quality_name=quality,
    )


def _encode(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"input", "output"},
        optional={
            "overwrite",
            "quality",
            "rotation_bits",
            "translation_bits",
            "rotation_error_degrees",
            "translation_error_fraction",
            "zlib_level",
            "format_version",
            "min_segment_frames",
            "max_segment_frames",
        },
        label="arguments",
    )
    source = _relative_path(workspace, arguments["input"], "arguments.input", input_file=True)
    output = _output_path(workspace, arguments)
    suffix = source.suffix.casefold()
    if suffix == ".bvh":
        clip = read_bvh(source)
        input_format = "bvh"
        metric_scope = "source_skeleton"
    elif suffix == ".npz":
        clip = read_smpl_npz(source)
        input_format = "smplx_parameter_stream"
        metric_scope = "rotation_and_root_translation_exact_global_position_proxy"
    else:
        raise AgentRequestError(
            "arguments.input must be BVH or a numeric SMPL-X NPZ motion"
        )
    settings = _codec_settings(arguments)
    format_version = int(
        _number(arguments.get("format_version", 1), "arguments.format_version", int)
    )
    if format_version not in (1, 2):
        raise AgentRequestError("arguments.format_version must be 1 or 2")
    has_segment_options = any(
        name in arguments for name in ("min_segment_frames", "max_segment_frames")
    )
    if format_version == 1 and has_segment_options:
        raise AgentRequestError("segment frame options require format_version 2")
    if format_version == 2:
        min_segment_frames = int(
            _number(
                arguments.get("min_segment_frames", 8),
                "arguments.min_segment_frames",
                int,
            )
        )
        max_segment_frames = int(
            _number(
                arguments.get("max_segment_frames", 32),
                "arguments.max_segment_frames",
                int,
            )
        )
        encoded = encode_v2_bytes(
            clip,
            settings,
            min_segment_frames=min_segment_frames,
            max_segment_frames=max_segment_frames,
        )
    else:
        encoded = encode_bytes(clip, settings)
    decoded = decode_bytes(encoded)
    _atomic_write(output, lambda path: path.write_bytes(encoded))
    return {
        "artifact": {
            "path": _display_path(workspace, output),
            "media_type": "application/vnd.skac.animation",
            "bytes": len(encoded),
        },
        "quality": settings.quality_name,
        "input_format": input_format,
        "metric_scope": metric_scope,
        "format_version": format_version,
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


def _inspect(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(arguments, required={"input"}, optional=set(), label="arguments")
    source = _relative_path(workspace, arguments["input"], "arguments.input", input_file=True)
    return inspect_file(source)


def _pack_create(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"entries", "output"},
        optional={"overwrite"},
        label="arguments",
    )
    supplied_entries = _object(arguments["entries"], "arguments.entries")
    if not supplied_entries:
        raise AgentRequestError("arguments.entries cannot be empty")
    entries: dict[str, bytes] = {}
    for entry_id, supplied_path in supplied_entries.items():
        source = _relative_path(
            workspace,
            supplied_path,
            f"arguments.entries.{entry_id}",
            input_file=True,
        )
        entries[entry_id] = source.read_bytes()
    output = _output_path(workspace, arguments)
    _atomic_write(output, lambda path: write_pack(path, entries))
    return {
        "artifact": {
            "path": _display_path(workspace, output),
            "media_type": "application/vnd.skac.pack",
            "bytes": output.stat().st_size,
        },
        **inspect_pack_file(output),
    }


def _pack_inspect(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(arguments, required={"input"}, optional=set(), label="arguments")
    source = _relative_path(
        workspace, arguments["input"], "arguments.input", input_file=True
    )
    return inspect_pack_file(source)


def _pack_extract(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"input", "entry", "output"},
        optional={"overwrite"},
        label="arguments",
    )
    source = _relative_path(
        workspace, arguments["input"], "arguments.input", input_file=True
    )
    entry_id = _string(arguments["entry"], "arguments.entry")
    output = _output_path(workspace, arguments)
    encoded = read_pack(source).entry_bytes(entry_id)
    _atomic_write(output, lambda path: path.write_bytes(encoded))
    return {
        "artifact": {
            "path": _display_path(workspace, output),
            "media_type": "application/vnd.skac.animation",
            "bytes": len(encoded),
        },
        "entry": entry_id,
    }


def _adaptive_plan(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"source", "report", "visual"},
        optional={
            "overwrite",
            "quality",
            "min_segment_frames",
            "max_segment_frames",
        },
        label="arguments",
    )
    source = _relative_path(
        workspace, arguments["source"], "arguments.source", input_file=True
    )
    overwrite = _boolean(arguments.get("overwrite", False), "arguments.overwrite")
    outputs: dict[str, Path] = {}
    for name in ("report", "visual"):
        output = _relative_path(
            workspace, arguments[name], f"arguments.{name}", input_file=False
        )
        if output.exists() and not overwrite:
            raise AgentRequestError(
                f"arguments.{name} already exists; set overwrite to true"
            )
        if output.exists() and not output.is_file():
            raise AgentRequestError(f"arguments.{name} is not a regular file")
        output.parent.mkdir(parents=True, exist_ok=True)
        outputs[name] = output
    if outputs["report"] == outputs["visual"]:
        raise AgentRequestError(
            "arguments.report and arguments.visual must be different files"
        )

    def integer(name: str, fallback: int) -> int:
        return (
            int(_number(arguments[name], f"arguments.{name}", int))
            if name in arguments
            else fallback
        )

    report, _ = build_adaptive_plan(
        read_bvh(source),
        settings=CodecSettings.preset(
            _string(arguments.get("quality", "high"), "arguments.quality")
        ),
        min_segment_frames=integer("min_segment_frames", 8),
        max_segment_frames=integer("max_segment_frames", 32),
    )
    _atomic_write(
        outputs["report"], lambda path: write_adaptive_plan_json(path, report)
    )
    _atomic_write(
        outputs["visual"], lambda path: write_adaptive_plan_svg(path, report)
    )
    return {
        "passed": report["passed"],
        "plan_sha256": report["plan_sha256"],
        "report": _display_path(workspace, outputs["report"]),
        "visual": _display_path(workspace, outputs["visual"]),
        "summary": report["summary"],
        "failed_checks": [
            item["id"] for item in report["checks"] if not item["passed"]
        ],
    }


def _decode(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"input", "output"},
        optional={"overwrite", "target", "profile"},
        label="arguments",
    )
    if ("target" in arguments) != ("profile" in arguments):
        raise AgentRequestError("arguments.target and arguments.profile must be supplied together")
    source = _relative_path(workspace, arguments["input"], "arguments.input", input_file=True)
    output = _output_path(workspace, arguments)
    clip = read_skac(source)
    diagnostics: dict[str, Any] = {}
    retargeted = False
    if "target" in arguments:
        target = _relative_path(
            workspace, arguments["target"], "arguments.target", input_file=True
        )
        profile_path = _relative_path(
            workspace, arguments["profile"], "arguments.profile", input_file=True
        )
        clip, diagnostics = retarget_motion(
            clip, read_bvh(target).skeleton, load_retarget_profile(profile_path)
        )
        retargeted = True
    _atomic_write(output, lambda path: write_bvh(path, clip))
    return {
        "artifact": {
            "path": _display_path(workspace, output),
            "media_type": "application/x-bvh",
            "bytes": output.stat().st_size,
        },
        "retargeted": retargeted,
        "frame_count": clip.frame_count,
        "joint_count": clip.skeleton.joint_count,
        "frame_time": clip.frame_time,
        "skeleton_sha256": clip.skeleton.signature(),
        **diagnostics,
    }


def _profile(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"source", "target", "output"},
        optional={"overwrite", "up_axis", "contact_lock"},
        label="arguments",
    )
    source = _relative_path(workspace, arguments["source"], "arguments.source", input_file=True)
    target = _relative_path(workspace, arguments["target"], "arguments.target", input_file=True)
    output = _output_path(workspace, arguments)
    up_axis_name = _string(arguments.get("up_axis", "Y"), "arguments.up_axis").upper()
    if up_axis_name not in {"X", "Y", "Z"}:
        raise AgentRequestError("arguments.up_axis must be X, Y, or Z")
    contact_lock = _boolean(arguments.get("contact_lock", False), "arguments.contact_lock")
    profile = build_retarget_profile(
        read_skac(source).skeleton,
        read_bvh(target).skeleton,
        up_axis={"X": 0, "Y": 1, "Z": 2}[up_axis_name],
        contact_lock=contact_lock,
    )
    _atomic_write(output, lambda path: save_retarget_profile(path, profile))
    return {
        "artifact": {
            "path": _display_path(workspace, output),
            "media_type": "application/vnd.skac.retarget-profile+json",
            "bytes": output.stat().st_size,
        },
        "profile_sha256": profile.signature(),
        "profile_schema_version": profile.schema_version,
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


def _runtime_skeleton(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _strict_fields(
        arguments,
        required={"target", "output"},
        optional={"overwrite"},
        label="arguments",
    )
    target = _relative_path(workspace, arguments["target"], "arguments.target", input_file=True)
    output = _output_path(workspace, arguments)
    skeleton = read_bvh(target).skeleton
    _atomic_write(output, lambda path: save_runtime_skeleton(path, skeleton))
    return {
        "artifact": {
            "path": _display_path(workspace, output),
            "media_type": "application/vnd.skac.runtime-skeleton+json",
            "bytes": output.stat().st_size,
        },
        "joint_count": skeleton.joint_count,
        "skeleton_sha256": skeleton.signature(),
    }


def _quality_gate_impl(
    workspace: Path,
    arguments: dict[str, Any],
    evaluation_case: str,
) -> dict[str, Any]:
    codec_optional = {
        "overwrite",
        "quality",
        "rotation_bits",
        "translation_bits",
        "rotation_error_degrees",
        "translation_error_fraction",
        "zlib_level",
        "minimum_decode_realtime_factor",
        "decode_iterations",
        "format_version",
        "min_segment_frames",
        "max_segment_frames",
    }
    different_optional = codec_optional | {
        "minimum_core_coverage",
        "maximum_retarget_frame_ms",
        "minimum_pipeline_realtime_factor",
        "pipeline_iterations",
        "frame_samples",
    }
    same_character = evaluation_case == "same_character"
    _strict_fields(
        arguments,
        required=(
            {"source", "report", "visual"}
            if same_character
            else {"source", "target", "report", "visual"}
        ),
        optional=codec_optional if same_character else different_optional,
        label="arguments",
    )
    source_path = _relative_path(
        workspace, arguments["source"], "arguments.source", input_file=True
    )
    target_path = None
    if not same_character:
        target_path = _relative_path(
            workspace, arguments["target"], "arguments.target", input_file=True
        )
    overwrite = _boolean(arguments.get("overwrite", False), "arguments.overwrite")
    outputs: dict[str, Path] = {}
    for name in ("report", "visual"):
        output = _relative_path(
            workspace, arguments[name], f"arguments.{name}", input_file=False
        )
        if output.exists() and not overwrite:
            raise AgentRequestError(f"arguments.{name} already exists; set overwrite to true")
        if output.exists() and not output.is_file():
            raise AgentRequestError(f"arguments.{name} is not a regular file")
        output.parent.mkdir(parents=True, exist_ok=True)
        outputs[name] = output
    if outputs["report"] == outputs["visual"]:
        raise AgentRequestError("arguments.report and arguments.visual must be different files")

    settings = _codec_settings(arguments)
    thresholds = QualityThresholds.for_settings(settings)
    format_version = int(
        _number(arguments.get("format_version", 1), "arguments.format_version", int)
    )
    if format_version not in (1, 2):
        raise AgentRequestError("arguments.format_version must be 1 or 2")
    if format_version == 1 and any(
        name in arguments for name in ("min_segment_frames", "max_segment_frames")
    ):
        raise AgentRequestError("segment frame options require format_version 2")

    def floating(name: str, fallback: float) -> float:
        return (
            float(_number(arguments[name], f"arguments.{name}", float))
            if name in arguments
            else fallback
        )

    def integer(name: str, fallback: int) -> int:
        return (
            int(_number(arguments[name], f"arguments.{name}", int))
            if name in arguments
            else fallback
        )

    thresholds = replace(
        thresholds,
        minimum_core_coverage=floating(
            "minimum_core_coverage", thresholds.minimum_core_coverage
        ),
        maximum_retarget_frame_ms_p95=floating(
            "maximum_retarget_frame_ms", thresholds.maximum_retarget_frame_ms_p95
        ),
        minimum_decode_realtime_factor=floating(
            "minimum_decode_realtime_factor", thresholds.minimum_decode_realtime_factor
        ),
        minimum_pipeline_realtime_factor=floating(
            "minimum_pipeline_realtime_factor", thresholds.minimum_pipeline_realtime_factor
        ),
    )
    source = read_bvh(source_path)
    target_skeleton = (
        source.skeleton if target_path is None else read_bvh(target_path).skeleton
    )
    report = run_quality_gate(
        source,
        target_skeleton,
        settings=settings,
        thresholds=thresholds,
        decode_iterations=integer("decode_iterations", 5),
        pipeline_iterations=integer("pipeline_iterations", 3),
        frame_samples=integer("frame_samples", 300),
        evaluation_case=evaluation_case,
        format_version=format_version,
        min_segment_frames=integer("min_segment_frames", 8),
        max_segment_frames=integer("max_segment_frames", 32),
    )
    write_quality_report_json(outputs["report"], report)
    write_quality_report_svg(outputs["visual"], report)
    return {
        "passed": report["passed"],
        "evaluation_case": report["evaluation_case"],
        "failed_checks": [item["id"] for item in report["checks"] if not item["passed"]],
        "report": _display_path(workspace, outputs["report"]),
        "visual": _display_path(workspace, outputs["visual"]),
        "performance": report["performance"],
    }


def _quality_gate_same(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    return _quality_gate_impl(workspace, arguments, "same_character")


def _quality_gate_different(
    workspace: Path, arguments: dict[str, Any]
) -> dict[str, Any]:
    return _quality_gate_impl(workspace, arguments, "different_character")


def _quality_gate(workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    return _quality_gate_impl(workspace, arguments, "auto")


_OPERATIONS: dict[str, Callable[[Path, dict[str, Any]], dict[str, Any]]] = {
    "encode": _encode,
    "inspect": _inspect,
    "pack_create": _pack_create,
    "pack_inspect": _pack_inspect,
    "pack_extract": _pack_extract,
    "adaptive_plan": _adaptive_plan,
    "decode": _decode,
    "profile": _profile,
    "runtime_skeleton": _runtime_skeleton,
    "quality_gate_same": _quality_gate_same,
    "quality_gate_different": _quality_gate_different,
    "quality_gate": _quality_gate,
}


def execute_request(value: Any, workspace: Path) -> tuple[dict[str, Any], int]:
    request_id: Any = None
    operation: Any = None
    try:
        request = _object(value, "request")
        request_id = request.get("request_id")
        operation = request.get("operation")
        _strict_fields(
            request,
            required={"protocol", "request_id", "operation", "arguments"},
            optional=set(),
            label="request",
        )
        if request["protocol"] != PROTOCOL:
            raise AgentRequestError(f"request.protocol must be {PROTOCOL}")
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            raise AgentRequestError("request.request_id must be a string or integer")
        operation = _string(operation, "request.operation")
        if operation not in _OPERATIONS:
            raise AgentRequestError(f"request.operation is not supported: {operation}")
        arguments = _object(request["arguments"], "request.arguments")
        result = _OPERATIONS[operation](workspace, arguments)
        return {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "operation": operation,
            "ok": True,
            "result": result,
        }, 0
    except AgentRequestError as error:
        return {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "operation": operation if isinstance(operation, str) else None,
            "ok": False,
            "error": {"code": "invalid_request", "message": str(error)},
        }, 2
    except Exception as error:  # The CLI boundary must never print a traceback to stdout.
        message = str(error).replace(str(workspace), ".")
        return {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "operation": operation if isinstance(operation, str) else None,
            "ok": False,
            "error": {
                "code": "operation_failed",
                "message": f"{type(error).__name__}: {message}",
            },
        }, 3


def _read_request_stream(path: str) -> str:
    if path == "-":
        payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    else:
        request_path = Path(path)
        if not request_path.is_file():
            raise AgentRequestError("request input does not name an existing file")
        if request_path.stat().st_size > MAX_REQUEST_BYTES:
            raise AgentRequestError(f"request input exceeds {MAX_REQUEST_BYTES} bytes")
        payload = request_path.read_bytes()
    if len(payload) > MAX_REQUEST_BYTES:
        raise AgentRequestError(f"request input exceeds {MAX_REQUEST_BYTES} bytes")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AgentRequestError("request input must be UTF-8") from error


def _emit(value: Any, *, pretty: bool = False) -> None:
    print(json.dumps(value, indent=2 if pretty else None, sort_keys=True))


def _run(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        _emit(
            {
                "protocol": PROTOCOL,
                "request_id": None,
                "operation": None,
                "ok": False,
                "error": {"code": "invalid_request", "message": "workspace is not a directory"},
            }
        )
        return 2
    try:
        text = _read_request_stream(args.request)
    except (AgentRequestError, OSError) as error:
        _emit(
            {
                "protocol": PROTOCOL,
                "request_id": None,
                "operation": None,
                "ok": False,
                "error": {"code": "invalid_request", "message": str(error)},
            }
        )
        return 2

    if not args.jsonl:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            response, status = execute_request({}, workspace)
            response["error"]["message"] = f"invalid JSON at line {error.lineno}, column {error.colno}"
            _emit(response, pretty=args.pretty)
            return status
        response, status = execute_request(value, workspace)
        _emit(response, pretty=args.pretty)
        return status

    status = 0
    saw_request = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        saw_request = True
        try:
            value = json.loads(line)
            response, request_status = execute_request(value, workspace)
        except json.JSONDecodeError as error:
            response = {
                "protocol": PROTOCOL,
                "request_id": None,
                "operation": None,
                "ok": False,
                "error": {
                    "code": "invalid_request",
                    "message": f"invalid JSON on JSONL line {line_number}, column {error.colno}",
                },
            }
            request_status = 2
        _emit(response)
        status = max(status, request_status)
    if not saw_request:
        _emit(
            {
                "protocol": PROTOCOL,
                "request_id": None,
                "operation": None,
                "ok": False,
                "error": {"code": "invalid_request", "message": "request input is empty"},
            }
        )
        return 2
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skac-agent", description="Machine-readable Agent interface for SKAC."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capability_parser = subparsers.add_parser(
        "capabilities", help="print the supported Agent protocol as JSON"
    )
    capability_parser.add_argument("--pretty", action="store_true")
    capability_parser.set_defaults(
        handler=lambda args: (_emit(capabilities(), pretty=args.pretty), 0)[1]
    )

    run_parser = subparsers.add_parser("run", help="execute a JSON or JSONL request")
    run_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    run_parser.add_argument("--request", default="-", help="UTF-8 request file, or - for stdin")
    run_parser.add_argument("--jsonl", action="store_true", help="process one request per line")
    run_parser.add_argument("--pretty", action="store_true", help="pretty-print a single response")
    run_parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run" and args.jsonl and args.pretty:
        build_parser().error("--pretty cannot be used with --jsonl")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
