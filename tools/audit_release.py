from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterator


TEXT_SUFFIXES = {
    ".cff", ".cfg", ".csv", ".html", ".ini", ".json", ".jsonl", ".md",
    ".py", ".toml", ".tsv", ".txt", ".xml", ".yaml", ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "LICENSE"}
ALLOWED_TOP_LEVEL = {
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "FORMAT.md",
    "LICENSE",
    "manifests",
    "PROTOCOL.md",
    "pyproject.toml",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "RETARGETING.md",
    "reports",
    "SECURITY.md",
    "skac_benchmark",
    "skac_codec",
    "skac_public_core",
    "tests",
    "THIRD_PARTY_NOTICES.md",
    "tools",
}
SKIPPED_METADATA_DIRS = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".svn",
    "__pycache__",
}
FORBIDDEN_DIRECTORY_NAMES = {
    "checkpoints", "data", "datasets", "downloads", "exports", "models",
    "outputs", "pretrained", "public_data", "third_party", "vendor",
}
FORBIDDEN_SUFFIXES = {
    ".7z", ".abc", ".avi", ".blend", ".bvh", ".ckpt", ".dll", ".dylib",
    ".exe", ".fbx", ".glb", ".gltf", ".gz", ".mkv", ".mov", ".mp4",
    ".npy", ".npz", ".obj", ".onnx", ".pb", ".pt", ".pth", ".rar",
    ".safetensors", ".skac", ".so", ".tar", ".tgz", ".usd", ".usda", ".usdc",
    ".usdz", ".zip",
}
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    re.compile(r"(?:^|[\s\"'])/(?:home|Users|mnt|opt|srv|var)/"),
)


def _deny_terms(args: argparse.Namespace) -> list[str]:
    terms: list[str] = []
    if args.denylist_file:
        terms.extend(
            line.strip()
            for line in args.denylist_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    terms.extend(term.strip() for term in os.environ.get("SKAC_PRIVATE_DENYLIST", "").split(","))
    return sorted({term.casefold() for term in terms if term})


def _contains_identifier(text: str, term: str) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])")
    return pattern.search(text) is not None


def _walk_release_tree(root: Path) -> Iterator[Path]:
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            path = current_path / name
            if name in SKIPPED_METADATA_DIRS:
                continue
            yield path
            if not path.is_symlink():
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(filenames):
            yield current_path / name


def audit(root: Path, deny_terms: list[str]) -> list[str]:
    findings: list[str] = []
    if not root.is_dir():
        return [f"release root is not a directory: {root}"]

    for child in root.iterdir():
        if child.name in SKIPPED_METADATA_DIRS:
            continue
        if child.name not in ALLOWED_TOP_LEVEL:
            findings.append(f"unexpected top-level entry: {child.name}")

    for path in _walk_release_tree(root):
        relative = path.relative_to(root).as_posix()
        folded_path = relative.casefold()

        for term in deny_terms:
            if _contains_identifier(folded_path, term):
                findings.append(f"denylisted term in path: {relative}")

        if path.is_symlink():
            findings.append(f"symbolic link is not allowed: {relative}")
            continue

        if path.is_dir():
            if path.name.casefold() in FORBIDDEN_DIRECTORY_NAMES:
                findings.append(f"release-excluded directory: {relative}")
            continue

        if not path.is_file():
            findings.append(f"unsupported filesystem entry: {relative}")
            continue

        suffix = path.suffix.casefold()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"release-excluded file type: {relative}")
            continue
        if path.name not in TEXT_FILENAMES and suffix not in TEXT_SUFFIXES:
            findings.append(f"unreviewed file type: {relative}")
            continue

        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            findings.append(f"file exceeds {MAX_PUBLIC_FILE_BYTES} bytes: {relative}")
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            findings.append(f"text file is not valid UTF-8: {relative}")
            continue

        folded_text = text.casefold()
        for term in deny_terms:
            if _contains_identifier(folded_text, term):
                findings.append(f"denylisted term in file: {relative}")
        if path.name != "audit_release.py":
            for pattern in ABSOLUTE_PATH_PATTERNS:
                if pattern.search(text):
                    findings.append(f"absolute filesystem path in file: {relative}")
                    break

    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a standalone public release tree before it is committed."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--denylist-file", type=Path)
    parser.add_argument("--require-private-denylist", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    terms = _deny_terms(args)
    if args.require_private_denylist and not terms:
        raise SystemExit("release audit failed: private denylist was not supplied")
    findings = audit(root, terms)
    if findings:
        details = "\n".join(f"- {finding}" for finding in findings)
        raise SystemExit(f"release audit failed:\n{details}")
    print(f"release audit passed: {root.name} ({len(terms)} private terms checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
