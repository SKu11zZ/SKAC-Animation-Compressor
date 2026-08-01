from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator


SKIPPED_METADATA_DIRS = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".svn",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic SHA-256 manifest for a release tree."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path, pattern: str, output: Path) -> Iterator[Path]:
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if name not in SKIPPED_METADATA_DIRS
        )
        for name in sorted(filenames):
            path = current_path / name
            if path.resolve() == output:
                continue
            relative = path.relative_to(root)
            if relative.match(pattern):
                yield path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    entries: list[dict[str, str | int]] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(iter_files(root, args.pattern, output))
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"refusing to hash symbolic link: {path.relative_to(root)}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        entries.append({"path": relative, "size": size, "sha256": digest})
        total_bytes += size
        tree_digest.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))

    report = {
        "algorithm": "sha256(relative_path + NUL + size + NUL + file_sha256 + LF)",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
