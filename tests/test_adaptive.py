from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skac_codec.adaptive import (
    adaptive_segments,
    build_adaptive_plan,
    joint_perceptual_importance,
    write_adaptive_plan_svg,
)
from skac_codec.format import CodecSettings
from skac_codec.math3d import euler_order_to_quaternion
from skac_codec.model import MotionClip, Skeleton


def adaptive_clip(frame_count: int = 96) -> MotionClip:
    skeleton = Skeleton(
        names=("Root", "Spine", "Hand", "Finger"),
        parents=np.asarray([-1, 0, 1, 2], dtype=np.int32),
        offsets=np.asarray(
            [[0, 0, 0], [0, 10, 0], [7, 8, 0], [4, 0, 0]],
            dtype=np.float64,
        ),
        channels=(
            (
                "Xposition",
                "Yposition",
                "Zposition",
                "Zrotation",
                "Xrotation",
                "Yrotation",
            ),
            ("Zrotation", "Xrotation", "Yrotation"),
            ("Zrotation", "Xrotation", "Yrotation"),
            ("Zrotation", "Xrotation", "Yrotation"),
        ),
        end_site_offsets=np.asarray(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.float64
        ),
        has_end_sites=np.asarray([False, False, False, True]),
    )
    phase = np.linspace(0.0, 4.0 * np.pi, frame_count)
    rotations = np.zeros((frame_count, skeleton.joint_count, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    rotations[:, 0] = euler_order_to_quaternion(
        "ZXY", np.stack((0.04 * np.sin(phase), phase * 0.0, phase * 0.0), axis=1)
    )
    rotations[:, 1] = euler_order_to_quaternion(
        "ZXY", np.stack((0.2 * np.sin(phase), 0.1 * np.cos(phase), phase * 0.0), axis=1)
    )
    hand_angle = 0.5 * np.sin(phase)
    hand_angle[frame_count // 2 :] += 0.35
    rotations[:, 2] = euler_order_to_quaternion(
        "ZXY", np.stack((hand_angle, phase * 0.0, 0.1 * np.cos(phase)), axis=1)
    )
    rotations[:, 3] = euler_order_to_quaternion(
        "ZXY", np.stack((0.15 * np.sin(2.0 * phase), phase * 0.0, phase * 0.0), axis=1)
    )
    translations = np.broadcast_to(
        skeleton.offsets, (frame_count, skeleton.joint_count, 3)
    ).copy()
    translations[:, 0, 0] = np.linspace(0.0, 20.0, frame_count)
    return MotionClip(skeleton, rotations, translations, 1.0 / 30.0)


class AdaptivePlanTests(unittest.TestCase):
    def test_plan_is_deterministic_segmented_and_quality_gated(self) -> None:
        source = adaptive_clip()
        settings = CodecSettings.preset("high")
        first, reconstructed = build_adaptive_plan(
            source,
            settings=settings,
            min_segment_frames=8,
            max_segment_frames=24,
        )
        second, _ = build_adaptive_plan(
            source,
            settings=settings,
            min_segment_frames=8,
            max_segment_frames=24,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["passed"])
        self.assertEqual(reconstructed.frame_count, source.frame_count)
        self.assertGreater(first["summary"]["segment_count"], 1)
        self.assertGreaterEqual(first["summary"]["rotation_budget_attempts"], 1)
        self.assertLessEqual(first["summary"]["rotation_budget_scale"], 1.0)
        self.assertTrue(all(item["passed"] for item in first["checks"]))
        self.assertTrue(
            all(
                item["reconstructed_error_degrees_max"]
                <= item["error_budget_degrees"] + 1e-12
                for item in first["rotation_tracks"]
            )
        )
        self.assertTrue(first["translation_tracks"])
        self.assertTrue(
            all(
                item["reconstructed_error_max"] <= item["error_budget"] + 1e-12
                for item in first["translation_tracks"]
            )
        )
        self.assertGreater(first["summary"]["planned_translation_key_count"], 0)
        self.assertGreater(
            len({item["bits"] for item in first["rotation_tracks"]}), 1
        )
        root_budgets = {
            item["error_budget_degrees"]
            for item in first["rotation_tracks"]
            if item["joint_name"] == "Root"
        }
        finger_budgets = {
            item["error_budget_degrees"]
            for item in first["rotation_tracks"]
            if item["joint_name"] == "Finger"
        }
        self.assertLess(max(root_budgets), min(finger_budgets))
        self.assertTrue(
            all(
                8 <= item["frame_count"] <= 24
                for item in first["segmentation"]["segments"]
            )
        )

    def test_importance_follows_hierarchy_and_svg_is_static_bilingual(self) -> None:
        source = adaptive_clip()
        importance = joint_perceptual_importance(source.skeleton)
        self.assertEqual(importance[0], 1.0)
        self.assertGreater(importance[1], importance[3])
        self.assertEqual(adaptive_segments(source, min_frames=8, max_frames=24)[0][0], 0)

        report, _ = build_adaptive_plan(
            source, min_segment_frames=8, max_segment_frames=24
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.svg"
            write_adaptive_plan_svg(path, report)
            svg = path.read_text(encoding="utf-8")
        self.assertIn("SKAC v2 Perceptual Plan", svg)
        self.assertIn("感知规划", svg)
        self.assertNotIn("<script", svg.casefold())
        self.assertNotIn("href=", svg.casefold())

    def test_checked_in_fixture_matches_the_planner(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report, _ = build_adaptive_plan(
            adaptive_clip(), min_segment_frames=8, max_segment_frames=24
        )
        stored = json.loads(
            (root / "reports" / "adaptive_plan_beta_fixture.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored, report)
        with tempfile.TemporaryDirectory() as temporary:
            rendered = Path(temporary) / "plan.svg"
            write_adaptive_plan_svg(rendered, report)
            self.assertEqual(
                rendered.read_text(encoding="utf-8"),
                (root / "reports" / "adaptive_plan_beta_fixture.svg").read_text(
                    encoding="utf-8"
                ),
            )


if __name__ == "__main__":
    unittest.main()
