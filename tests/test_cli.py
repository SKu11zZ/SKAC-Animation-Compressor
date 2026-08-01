import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skac_benchmark.cli import evaluate_manifest
from test_manifest import public_record


class EvaluatorTests(unittest.TestCase):
    def test_manifest_to_four_group_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "manifests"
            prediction_dir = root / "public_data" / "predictions"
            truth_dir = root / "public_data" / "ground_truth"
            manifest_dir.mkdir()
            prediction_dir.mkdir(parents=True)
            truth_dir.mkdir(parents=True)

            positions = np.zeros((2, 22, 3), dtype=np.float64)
            np.savez(prediction_dir / "sample_1.npz", positions=positions)
            np.savez(truth_dir / "sample_1.npz", positions=positions)
            manifest_path = manifest_dir / "samples.jsonl"
            manifest_path.write_text(json.dumps(public_record()) + "\n", encoding="utf-8")

            report = evaluate_manifest(manifest_path)
            self.assertEqual(report["quality_layer"], "retargeting")
            self.assertEqual(report["overall"]["sample_count"], 1)
            self.assertEqual(
                report["groups"]["seen_character_unseen_motion"]["sample_count"], 1
            )
            self.assertEqual(report["overall"]["global_mse_raw"], 0.0)


if __name__ == "__main__":
    unittest.main()
