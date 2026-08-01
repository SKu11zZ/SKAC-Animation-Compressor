from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "run_public_san_rotation_copy.py"
)
SPEC = importlib.util.spec_from_file_location("public_san_rotation_copy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PublicSanRotationCopyTests(unittest.TestCase):
    def test_canonical_joint_name_removes_namespace(self) -> None:
        self.assertEqual(MODULE.canonical_joint_name("character:Hips"), "Hips")
        self.assertEqual(MODULE.canonical_joint_name("LeftArm"), "LeftArm")

    def test_mapping_uses_public_semantic_names(self) -> None:
        mapping = MODULE.build_index_mapping(
            ["source:Hips", "source:Spine", "source:Head"],
            ["Hips", "Spine", "SpineExtra", "Head"],
        )
        self.assertEqual(mapping, [(0, 0), (1, 1), (3, 2)])


if __name__ == "__main__":
    unittest.main()
