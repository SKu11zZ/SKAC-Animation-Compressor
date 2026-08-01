"""Blender-only background bridge. It is launched by skac_codec.fbx."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


def _arguments() -> argparse.Namespace:
    try:
        separator = sys.argv.index("--")
    except ValueError as error:
        raise RuntimeError("bridge arguments require Blender's -- separator") from error
    parser = argparse.ArgumentParser(prog="skac-blender-fbx-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--armature")

    inject = subparsers.add_parser("inject")
    inject.add_argument("--template", type=Path, required=True)
    inject.add_argument("--animation", type=Path, required=True)
    inject.add_argument("--output", type=Path, required=True)
    inject.add_argument("--armature")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)

    for subparser in (extract, inject, validate):
        subparser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[separator + 1 :])


def _write_report(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _enable_bvh_operators() -> None:
    if hasattr(bpy.ops.import_anim, "bvh") and hasattr(bpy.ops.export_anim, "bvh"):
        return
    try:
        bpy.ops.preferences.addon_enable(module="io_anim_bvh")
    except Exception as error:
        raise RuntimeError(
            "Blender's BVH import/export add-on is required for the experimental FBX backend"
        ) from error
    if not hasattr(bpy.ops.import_anim, "bvh") or not hasattr(bpy.ops.export_anim, "bvh"):
        raise RuntimeError("Blender BVH import/export operators are unavailable")


def _import_fbx(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = bpy.ops.import_scene.fbx(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError("Blender FBX importer did not finish")


def _armatures() -> list[bpy.types.Object]:
    return [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]


def _choose_armature(name: str | None) -> bpy.types.Object:
    armatures = _armatures()
    if name is not None:
        matches = [item for item in armatures if item.name == name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one armature named {name!r}, found {len(matches)}")
        return matches[0]
    if not armatures:
        raise RuntimeError("FBX scene contains no armature")
    animated = [
        item
        for item in armatures
        if item.animation_data is not None and item.animation_data.action is not None
    ]
    candidates = animated or armatures
    return max(candidates, key=lambda item: len(item.data.bones))


def _frame_range(armature: bpy.types.Object) -> tuple[int, int]:
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        frame = int(bpy.context.scene.frame_current)
        return frame, frame
    start, end = action.frame_range
    return int(round(start)), int(round(end))


def _select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _scene_summary() -> dict[str, object]:
    armatures = _armatures()
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    materials = {slot.material.name for item in meshes for slot in item.material_slots if slot.material}
    actions = list(bpy.data.actions)
    keyframe_count = sum(
        len(curve.keyframe_points) for action in actions for curve in action.fcurves
    )
    missing_dependencies = sorted(
        image.name
        for image in bpy.data.images
        if image.source == "FILE"
        and image.packed_file is None
        and image.filepath
        and not Path(bpy.path.abspath(image.filepath)).is_file()
    )
    return {
        "armature_count": len(armatures),
        "bone_count": sum(len(item.data.bones) for item in armatures),
        "mesh_count": len(meshes),
        "material_count": len(materials),
        "action_count": len(actions),
        "keyframe_count": keyframe_count,
        "missing_external_dependencies": missing_dependencies,
    }


def _extract(args: argparse.Namespace) -> dict[str, object]:
    _reset_scene()
    _enable_bvh_operators()
    _import_fbx(args.input)
    armature = _choose_armature(args.armature)
    start, end = _frame_range(armature)
    _select_only(armature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_anim.bvh(
        filepath=str(args.output),
        frame_start=start,
        frame_end=end,
        root_transform_only=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError("Blender BVH exporter did not finish")
    return {
        "status": "ok",
        "operation": "extract",
        "armature_name": armature.name,
        "bone_count": len(armature.data.bones),
        "frame_start": start,
        "frame_end": end,
        **_scene_summary(),
    }


def _bone_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.rsplit(":", 1)[-1].casefold())


def _inject(args: argparse.Namespace) -> dict[str, object]:
    _reset_scene()
    _enable_bvh_operators()
    _import_fbx(args.template)
    target = _choose_armature(args.armature)
    expected = _scene_summary()
    target_name = target.name

    existing_objects = set(bpy.context.scene.objects)
    result = bpy.ops.import_anim.bvh(filepath=str(args.animation))
    if "FINISHED" not in result:
        raise RuntimeError("Blender BVH importer did not finish")
    imported = [
        item
        for item in bpy.context.scene.objects
        if item not in existing_objects and item.type == "ARMATURE"
    ]
    if len(imported) != 1:
        raise RuntimeError(f"expected one BVH driver armature, found {len(imported)}")
    driver = imported[0]
    start, end = _frame_range(driver)

    driver_bones = {_bone_key(item.name): item for item in driver.pose.bones}
    mapped = 0
    target.animation_data_clear()
    for target_bone in target.pose.bones:
        driver_bone = driver_bones.get(_bone_key(target_bone.name))
        if driver_bone is None:
            continue
        constraint = target_bone.constraints.new(type="COPY_TRANSFORMS")
        constraint.target = driver
        constraint.subtarget = driver_bone.name
        constraint.owner_space = "POSE"
        constraint.target_space = "POSE"
        mapped += 1
    if mapped == 0:
        raise RuntimeError("target FBX and animation BVH have no matching bones")

    _select_only(target)
    bpy.ops.object.mode_set(mode="POSE")
    for bone in target.data.bones:
        bone.select = True
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    bake_result = bpy.ops.nla.bake(
        frame_start=start,
        frame_end=end,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=True,
        clean_curves=False,
        bake_types={"POSE"},
    )
    if "FINISHED" not in bake_result:
        raise RuntimeError("Blender pose bake did not finish")
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.data.objects.remove(driver, do_unlink=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_result = bpy.ops.export_scene.fbx(
        filepath=str(args.output),
        use_selection=False,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        add_leaf_bones=False,
        bake_anim=True,
        path_mode="COPY",
        embed_textures=True,
    )
    if "FINISHED" not in export_result:
        raise RuntimeError("Blender FBX exporter did not finish")

    _reset_scene()
    _import_fbx(args.output)
    actual = _scene_summary()
    if int(actual["mesh_count"]) < int(expected["mesh_count"]):
        raise RuntimeError("validated FBX lost one or more meshes")
    if int(actual["material_count"]) < int(expected["material_count"]):
        raise RuntimeError("validated FBX lost one or more materials")
    if int(actual["bone_count"]) < int(expected["bone_count"]):
        raise RuntimeError("validated FBX lost one or more bones")
    if int(actual["keyframe_count"]) <= 0:
        raise RuntimeError("validated FBX contains no animation keyframes")
    if actual["missing_external_dependencies"]:
        raise RuntimeError("validated FBX has missing external dependencies")
    return {
        "status": "ok",
        "operation": "inject",
        "armature_name": target_name,
        "mapped_bone_count": mapped,
        "frame_start": start,
        "frame_end": end,
        "expected_mesh_count": expected["mesh_count"],
        "expected_material_count": expected["material_count"],
        **actual,
    }


def _validate(args: argparse.Namespace) -> dict[str, object]:
    _reset_scene()
    _import_fbx(args.input)
    return {"status": "ok", "operation": "validate", **_scene_summary()}


def main() -> None:
    args = _arguments()
    try:
        if args.command == "extract":
            report = _extract(args)
        elif args.command == "inject":
            report = _inject(args)
        else:
            report = _validate(args)
        _write_report(args.report, report)
    except Exception as error:
        _write_report(
            args.report,
            {"status": "error", "error_type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
