from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tools.run_v2_planner_population import (
    _aggregate,
    _representative_samples,
    _stratified_selection,
    _svg,
)


class V2PlannerPopulationTests(unittest.TestCase):
    def test_duration_stratified_selection_is_deterministic_and_unique(self) -> None:
        motions = [
            (f"smpl_{index:04d}", Path(f"motion_{index}.npz"), 100 + index * 7)
            for index in range(11)
        ]
        selected = _stratified_selection(list(reversed(motions)), 5)
        self.assertEqual([item[0] for item in selected], [
            "smpl_0000", "smpl_0002", "smpl_0005", "smpl_0008", "smpl_0010"
        ])

    def test_aggregate_reports_percentiles_coverage_and_static_svg(self) -> None:
        samples = []
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
            samples.append(
                {
                    "frame_count": 10,
                    "source_frame_count": 20,
                    "passed": index < 4,
                    "planner_ms": value,
                    "planned_payload_percent_of_float32": value,
                    "rotation_error_degrees_max": value / 100.0,
                    "global_position_error_max_height_fraction": value / 1000.0,
                    "rotation_bits_weighted_mean": value + 8.0,
                    "rotation_budget_attempts": 1.0,
                }
            )
        aggregate = _aggregate(samples)
        self.assertEqual(aggregate["quality_pass_coverage_percent"], 80.0)
        self.assertEqual(aggregate["distributions"]["planner_ms"]["p50"], 3.0)
        self.assertAlmostEqual(
            aggregate["distributions"]["planner_ms"]["p80"],
            float(np.percentile([1, 2, 3, 4, 5], 80)),
        )
        report = {"aggregate": aggregate}
        svg = _svg(report)
        self.assertIn("P95", svg)
        self.assertIn("人口分布校准", svg)
        self.assertNotIn("<script", svg.casefold())
        for index, sample in enumerate(samples):
            sample["id"] = f"sample_{index}"
        representatives = _representative_samples(samples, "planner_ms")
        self.assertEqual(
            [item["label"] for item in representatives],
            ["best", "p50", "p80", "p95", "worst"],
        )
        self.assertEqual(len({item["id"] for item in representatives}), 5)


if __name__ == "__main__":
    unittest.main()
