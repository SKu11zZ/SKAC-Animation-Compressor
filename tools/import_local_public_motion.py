from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


FAMILIES = {
    "mixamo": {".fbx"},
    "manny": {".fbx"},
    "smpl": {".npz", ".txt"},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy explicitly supplied public motion sources into a neutral, "
            "path-free local cache."
        )
    )
    parser.add_argument("--mixamo", type=Path, required=True)
    parser.add_argument("--manny", type=Path, required=True)
    parser.add_argument("--smpl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.is_dir():
        raise ValueError("an explicitly supplied public source directory is missing")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    ]
    if not files:
        raise ValueError("an explicitly supplied public source contains no supported files")
    return sorted(
        files,
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def import_cache(sources: dict[str, Path], output_root: Path) -> dict[str, Any]:
    if set(sources) != set(FAMILIES):
        raise ValueError("the local cache requires Mixamo, Manny, and SMPL sources")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("the local public-motion cache already exists")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output_root.name}.importing"
    if staging.exists():
        raise FileExistsError("a local public-motion import is already in progress")
    staging.mkdir()

    family_records: list[dict[str, Any]] = []
    try:
        for family in ("mixamo", "manny", "smpl"):
            source_root = sources[family].resolve()
            files = _source_files(source_root, FAMILIES[family])
            family_root = staging / family
            family_root.mkdir()
            entries: list[dict[str, Any]] = []
            for index, source in enumerate(files, 1):
                identifier = f"{family}_{index:04d}"
                suffix = source.suffix.casefold()
                destination = family_root / f"{identifier}{suffix}"
                shutil.copyfile(source, destination)
                entries.append(
                    {
                        "id": identifier,
                        "relative_path": destination.relative_to(staging).as_posix(),
                        "format": suffix.removeprefix("."),
                        "bytes": destination.stat().st_size,
                        "sha256": _sha256(destination),
                    }
                )
            family_records.append(
                {
                    "family": family,
                    "file_count": len(entries),
                    "total_bytes": sum(int(item["bytes"]) for item in entries),
                    "files": entries,
                }
            )

        manifest: dict[str, Any] = {
            "schema": "skac.local_public_motion_cache",
            "schema_version": "1.0.0",
            "path_policy": "neutral_relative_paths_only",
            "families": family_records,
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            _canonical_json(manifest)
        ).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = import_cache(
        {
            "mixamo": args.mixamo,
            "manny": args.manny,
            "smpl": args.smpl,
        },
        args.output_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "families": [
                    {
                        "family": item["family"],
                        "file_count": item["file_count"],
                        "total_bytes": item["total_bytes"],
                    }
                    for item in manifest["families"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
