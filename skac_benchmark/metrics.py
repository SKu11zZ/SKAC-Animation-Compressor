from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MotionMetrics:
    global_mse_raw: float
    global_mse_x1e3: float
    local_mse_raw: float
    local_mse_x1e3: float
    san_paper_position_error_x1e3: float
    san_official_code_mse_raw: float
    san_official_code_mse_x1e3: float
    penetration_percent: float | None
    contact_distance_cm: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _positions(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, joints, 3)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalized_coordinate_mse(
    predicted: ArrayLike, ground_truth: ArrayLike, character_height: float
) -> float:
    pred = _positions(predicted, "predicted")
    truth = _positions(ground_truth, "ground_truth")
    if pred.shape != truth.shape:
        raise ValueError("predicted and ground_truth shapes must match")
    if not np.isfinite(character_height) or character_height <= 0:
        raise ValueError("character_height must be positive and finite")
    return float(np.mean(np.square(pred - truth)) / (character_height**2))


def root_aligned_positions(
    predicted: ArrayLike, ground_truth: ArrayLike, root_index: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pred = _positions(predicted, "predicted")
    truth = _positions(ground_truth, "ground_truth")
    if pred.shape != truth.shape:
        raise ValueError("predicted and ground_truth shapes must match")
    if not 0 <= root_index < pred.shape[1]:
        raise ValueError("root_index is outside the joint range")
    aligned = pred - pred[:, root_index : root_index + 1] + truth[:, root_index : root_index + 1]
    return aligned, truth


def san_paper_position_error_x1e3(
    predicted: ArrayLike, ground_truth: ArrayLike, character_height: float
) -> float:
    pred = _positions(predicted, "predicted")
    truth = _positions(ground_truth, "ground_truth")
    if pred.shape != truth.shape:
        raise ValueError("predicted and ground_truth shapes must match")
    if not np.isfinite(character_height) or character_height <= 0:
        raise ValueError("character_height must be positive and finite")
    distances = np.linalg.norm(pred - truth, axis=-1)
    return float(np.mean(distances) / character_height * 1_000.0)


def penetration_percent(penetrated_vertices: ArrayLike, total_vertices: ArrayLike) -> float:
    penetrated = np.asarray(penetrated_vertices, dtype=np.float64)
    total = np.asarray(total_vertices, dtype=np.float64)
    if penetrated.shape != total.shape or penetrated.ndim != 1 or penetrated.size == 0:
        raise ValueError("penetration counts must be non-empty matching per-frame vectors")
    if not np.isfinite(penetrated).all() or not np.isfinite(total).all():
        raise ValueError("penetration counts must be finite")
    if np.any(total <= 0) or np.any(penetrated < 0) or np.any(penetrated > total):
        raise ValueError("penetration counts must satisfy 0 <= penetrated <= total")
    return float(np.mean(penetrated / total) * 100.0)


def mean_contact_distance_cm(distances_cm: ArrayLike) -> float:
    distances = np.asarray(distances_cm, dtype=np.float64)
    if distances.size == 0 or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("contact distances must be non-empty, finite, and non-negative")
    return float(np.mean(distances))


def evaluate_motion(
    predicted: ArrayLike,
    ground_truth: ArrayLike,
    character_height: float,
    root_index: int,
    penetrated_vertices: ArrayLike | None = None,
    total_limb_vertices: ArrayLike | None = None,
    contact_distances_cm: ArrayLike | None = None,
) -> MotionMetrics:
    pred = _positions(predicted, "predicted")
    truth = _positions(ground_truth, "ground_truth")
    global_raw = normalized_coordinate_mse(pred, truth, character_height)
    aligned, aligned_truth = root_aligned_positions(pred, truth, root_index)
    local_raw = normalized_coordinate_mse(aligned, aligned_truth, character_height)

    if (penetrated_vertices is None) != (total_limb_vertices is None):
        raise ValueError("both penetration count arrays must be provided together")
    penetration = None
    if penetrated_vertices is not None and total_limb_vertices is not None:
        penetration = penetration_percent(penetrated_vertices, total_limb_vertices)

    contact = None
    if contact_distances_cm is not None:
        contact = mean_contact_distance_cm(contact_distances_cm)

    return MotionMetrics(
        global_mse_raw=global_raw,
        global_mse_x1e3=global_raw * 1_000.0,
        local_mse_raw=local_raw,
        local_mse_x1e3=local_raw * 1_000.0,
        san_paper_position_error_x1e3=san_paper_position_error_x1e3(
            pred, truth, character_height
        ),
        san_official_code_mse_raw=global_raw,
        san_official_code_mse_x1e3=global_raw * 1_000.0,
        penetration_percent=penetration,
        contact_distance_cm=contact,
    )
