from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NativeRuntimeBetaTests(unittest.TestCase):
    def test_c_abi_exposes_required_playback_operations(self) -> None:
        header = (ROOT / "native/include/skac_runtime.h").read_text(encoding="utf-8")
        for symbol in (
            "SKAC_RUNTIME_ABI_VERSION",
            "skac_decoder_open_container",
            "skac_decoder_open_memory",
            "skac_decoder_sample_frame",
            "skac_decoder_sample_time",
            "skac_decoder_get_joint_name",
        ):
            self.assertIn(symbol, header)

    def test_unity_package_is_beta_and_has_no_binary(self) -> None:
        package_root = ROOT / "integrations/unity/com.skac.runtime"
        manifest = json.loads((package_root / "package.json").read_text(encoding="utf-8"))
        self.assertIn("beta", manifest["version"])
        self.assertTrue((package_root / "Runtime/SkacRuntime.cs").is_file())
        binary_suffixes = {".dll", ".dylib", ".so"}
        self.assertFalse(
            any(path.suffix.casefold() in binary_suffixes for path in package_root.rglob("*"))
        )

    def test_unreal_plugin_uses_host_inflate_callback(self) -> None:
        plugin_root = ROOT / "integrations/unreal/SKACRuntimeBeta"
        plugin = json.loads((plugin_root / "SKACRuntimeBeta.uplugin").read_text(encoding="utf-8"))
        self.assertTrue(plugin["IsBetaVersion"])
        implementation = (plugin_root / "Source/SKACRuntimeBeta/Private/SkacClip.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("skac_decoder_open_container", implementation)
        self.assertIn("FCompression::UncompressMemory", implementation)

    def test_runtime_scope_is_documented_in_both_languages(self) -> None:
        document = (ROOT / "RUNTIME_BETA.md").read_text(encoding="utf-8")
        self.assertIn("## English", document)
        self.assertIn("## 中文", document)
        self.assertIn("whole-clip", document)
        self.assertIn("整段载入", document)


if __name__ == "__main__":
    unittest.main()
