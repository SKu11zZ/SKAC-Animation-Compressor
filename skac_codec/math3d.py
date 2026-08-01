from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_NEXT_AXIS = (1, 2, 0, 1)
_STATIC_AXES = {
    "XYZ": (0, 0),
    "XZY": (0, 1),
    "YXZ": (1, 1),
    "YZX": (1, 0),
    "ZXY": (2, 0),
    "ZYX": (2, 1),
}


def normalize_quaternions(value: ArrayLike) -> FloatArray:
    quaternions = np.asarray(value, dtype=np.float64)
    if quaternions.shape[-1] != 4:
        raise ValueError("quaternions require a final dimension of four")
    lengths = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12) or not np.isfinite(lengths).all():
        raise ValueError("invalid quaternion")
    return quaternions / lengths


def quaternion_multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def axis_angle_quaternion(axis: str, angle_radians: ArrayLike) -> FloatArray:
    angles = np.asarray(angle_radians, dtype=np.float64)
    half = angles * 0.5
    result = np.zeros(angles.shape + (4,), dtype=np.float64)
    result[..., 0] = np.cos(half)
    component = {"X": 1, "Y": 2, "Z": 3}.get(axis.upper())
    if component is None:
        raise ValueError(f"invalid rotation axis: {axis}")
    result[..., component] = np.sin(half)
    return result


def euler_order_to_quaternion(order: str, angles_radians: ArrayLike) -> FloatArray:
    order = order.upper()
    if order not in _STATIC_AXES:
        raise ValueError(f"unsupported Euler order: {order}")
    angles = np.asarray(angles_radians, dtype=np.float64)
    if angles.shape[-1] != 3:
        raise ValueError("Euler angles require a final dimension of three")
    result = np.zeros(angles.shape[:-1] + (4,), dtype=np.float64)
    result[..., 0] = 1.0
    for component, axis in enumerate(order):
        result = quaternion_multiply(
            result, axis_angle_quaternion(axis, angles[..., component])
        )
    return normalize_quaternions(result)


def quaternion_to_matrix(value: ArrayLike) -> FloatArray:
    q = normalize_quaternions(value)
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def quaternion_slerp(start: ArrayLike, end: ArrayLike, amount: ArrayLike) -> FloatArray:
    first = normalize_quaternions(start)
    second = normalize_quaternions(end)
    amount_array = np.asarray(amount, dtype=np.float64)
    first, second = np.broadcast_arrays(first, second)
    target_shape = np.broadcast_shapes(first.shape[:-1], amount_array.shape)
    first = np.broadcast_to(first, target_shape + (4,)).reshape(-1, 4)
    second = np.broadcast_to(second, target_shape + (4,)).reshape(-1, 4).copy()
    amount_array = np.broadcast_to(amount_array, target_shape).reshape(-1)

    dots = np.sum(first * second, axis=-1)
    negative = dots < 0.0
    second[negative] *= -1.0
    dots = np.abs(dots)
    linear = dots > 0.9995

    result = np.empty_like(first)
    if np.any(linear):
        weights = amount_array[linear, None]
        result[linear] = first[linear] + weights * (second[linear] - first[linear])
    if np.any(~linear):
        theta = np.arccos(np.clip(dots[~linear], -1.0, 1.0))
        sine = np.sin(theta)
        amount_values = amount_array[~linear]
        first_weight = np.sin((1.0 - amount_values) * theta) / sine
        second_weight = np.sin(amount_values * theta) / sine
        result[~linear] = (
            first_weight[:, None] * first[~linear]
            + second_weight[:, None] * second[~linear]
        )
    return normalize_quaternions(result).reshape(target_shape + (4,))


def _static_euler_from_matrix(matrix: FloatArray, order: str) -> tuple[float, float, float]:
    first_axis, parity = _STATIC_AXES[order]
    i = first_axis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    cy = math.sqrt(matrix[i, i] * matrix[i, i] + matrix[j, i] * matrix[j, i])
    if cy > np.finfo(np.float64).eps * 4.0:
        first = math.atan2(matrix[k, j], matrix[k, k])
        second = math.atan2(-matrix[k, i], cy)
        third = math.atan2(matrix[j, i], matrix[i, i])
    else:
        first = math.atan2(-matrix[j, k], matrix[j, j])
        second = math.atan2(-matrix[k, i], cy)
        third = 0.0
    if parity:
        first, second, third = -first, -second, -third
    return first, second, third


def quaternion_to_euler_order(value: ArrayLike, order: str) -> FloatArray:
    order = order.upper()
    if order not in _STATIC_AXES:
        raise ValueError(f"unsupported Euler order: {order}")
    matrices = quaternion_to_matrix(value)
    flat = matrices.reshape((-1, 3, 3))
    reverse_order = order[::-1]
    angles = np.asarray(
        [_static_euler_from_matrix(matrix, reverse_order)[::-1] for matrix in flat],
        dtype=np.float64,
    )
    return angles.reshape(matrices.shape[:-2] + (3,))


def quaternion_angular_error_degrees(left: ArrayLike, right: ArrayLike) -> FloatArray:
    a = normalize_quaternions(left)
    b = normalize_quaternions(right)
    dots = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
