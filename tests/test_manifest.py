import json
import tempfile
import unittest
from pathlib import Path

from skac_benchmark.manifest import (
    Sample,
    load_manifest,
    require_frozen_skeleton_configs,
    validate_record,
)


def public_record() -> dict:
    return {
        "schema_version": "0.1.0",
        "sample_id": "sample_1",
        "dataset": "Mixamo",
        "method": "SKAC",
        "quality_layer": "retargeting",
        "track": "automatic",
        "source_character_id": "character_1",
        "target_character_id": "character_2",
        "motion_id": "motion_1",
        "character_seen": True,
        "motion_seen": False,
        "joint_count": 22,
        "root_index": 0,
        "character_height": 1.8,
        "prediction_path": "public_data/predictions/sample_1.npz",
        "ground_truth_path": "public_data/ground_truth/sample_1.npz",
        "skeleton_config_id": "character_2_v1",
        "skeleton_config_sha256": "a" * 64,
        "configuration_scope": "skeleton",
        "per_animation_adjustment": False,
        "provenance_url": "https://www.mixamo.com/",
        "license_note": "Public access note",
        "geometry_backend": None,
    }


class ManifestTests(unittest.TestCase):
    def test_split_is_derived_from_frozen_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "manifest.jsonl"
            path.write_text(json.dumps(public_record()) + "\n", encoding="utf-8")
            sample = load_manifest(path)[0]
            self.assertEqual(sample.split, "seen_character_unseen_motion")

    def test_automatic_track_rejects_clip_specific_adjustment(self) -> None:
        record = public_record()
        record["per_animation_adjustment"] = True
        with self.assertRaisesRegex(ValueError, "forbids per-animation"):
            validate_record(record)

    def test_absolute_or_parent_path_is_rejected(self) -> None:
        for path in ("C:" + "/private/sample.npz", "../sample.npz"):
            record = public_record()
            record["prediction_path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_record(record)

    def test_target_skeleton_configuration_is_frozen(self) -> None:
        first = public_record()
        second = public_record()
        second["sample_id"] = "sample_2"
        second["motion_id"] = "motion_2"
        second["skeleton_config_sha256"] = "b" * 64
        samples = [Sample(first, Path(".")), Sample(second, Path("."))]
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            require_frozen_skeleton_configs(samples)


if __name__ == "__main__":
    unittest.main()
