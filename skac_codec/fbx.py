from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


class FbxBackendUnavailable(RuntimeError):
    pass


class FbxBridgeError(RuntimeError):
    pass


def find_blender(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if not candidate.is_file():
            raise FbxBackendUnavailable(f"Blender executable does not exist: {candidate}")
        return candidate
    located = shutil.which("blender")
    if located:
        return Path(located).resolve()
    raise FbxBackendUnavailable(
        "Blender was not found. Pass --blender with a Blender executable path."
    )


def _bridge_script() -> Path:
    return Path(__file__).with_name("_blender_fbx_bridge.py").resolve()


def _run_bridge(
    blender: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with tempfile.TemporaryDirectory(prefix="skac-fbx-report-") as temporary:
        report_path = Path(temporary) / "report.json"
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(_bridge_script()),
            "--",
            *arguments,
            "--report",
            str(report_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise FbxBridgeError(
                f"Blender FBX bridge failed with exit code {completed.returncode}: {details[-2000:]}"
            )
        if not report_path.is_file():
            raise FbxBridgeError("Blender FBX bridge did not create its report")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FbxBridgeError("Blender FBX bridge returned an invalid report") from error
        if not isinstance(report, dict) or report.get("status") != "ok":
            raise FbxBridgeError("Blender FBX bridge report did not confirm success")
        return report


def _temporary_output(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{final_path.stem}-",
        suffix=final_path.suffix,
        dir=final_path.parent,
        delete=False,
    )
    handle.close()
    path = Path(handle.name)
    path.unlink()
    return path


def extract_fbx_to_bvh(
    input_path: Path,
    output_path: Path,
    *,
    blender_executable: Path | None = None,
    armature_name: str | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    blender = find_blender(blender_executable)
    temporary_output = _temporary_output(output_path.resolve())
    arguments = ["extract", "--input", str(source), "--output", str(temporary_output)]
    if armature_name:
        arguments.extend(("--armature", armature_name))
    try:
        report = _run_bridge(blender, arguments, timeout_seconds=timeout_seconds)
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise FbxBridgeError("Blender did not create a non-empty BVH output")
        os.replace(temporary_output, output_path.resolve())
        return report
    finally:
        temporary_output.unlink(missing_ok=True)


def inject_bvh_into_fbx(
    template_path: Path,
    animation_path: Path,
    output_path: Path,
    *,
    blender_executable: Path | None = None,
    armature_name: str | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    template = template_path.resolve()
    animation = animation_path.resolve()
    if not template.is_file():
        raise FileNotFoundError(template)
    if not animation.is_file():
        raise FileNotFoundError(animation)
    blender = find_blender(blender_executable)
    temporary_output = _temporary_output(output_path.resolve())
    arguments = [
        "inject",
        "--template",
        str(template),
        "--animation",
        str(animation),
        "--output",
        str(temporary_output),
    ]
    if armature_name:
        arguments.extend(("--armature", armature_name))
    try:
        report = _run_bridge(blender, arguments, timeout_seconds=timeout_seconds)
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise FbxBridgeError("Blender did not create a non-empty FBX output")
        os.replace(temporary_output, output_path.resolve())
        return report
    finally:
        temporary_output.unlink(missing_ok=True)


def validate_fbx(
    input_path: Path,
    *,
    blender_executable: Path | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    blender = find_blender(blender_executable)
    report = _run_bridge(
        blender,
        ["validate", "--input", str(source)],
        timeout_seconds=timeout_seconds,
    )
    if int(report.get("armature_count", 0)) < 1:
        raise FbxBridgeError("FBX validation found no armature")
    return report
