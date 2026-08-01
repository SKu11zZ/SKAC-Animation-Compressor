from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from skac_codec.bvh import read_bvh
from skac_codec.runtime import save_runtime_skeleton


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a public BVH skeleton for the native SKAC Runtime Beta."
    )
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    save_runtime_skeleton(args.output, read_bvh(args.target).skeleton)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
