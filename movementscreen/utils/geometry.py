"""Geometric helpers for joint angle and positional calculations."""
from __future__ import annotations
import numpy as np


def angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Return the angle (degrees) at *vertex* formed by vectors vertex→a and vertex→b.

    Works in 2-D or 3-D. Input arrays should be shape (2,) or (3,).
    """
    va = a - vertex
    vb = b - vertex
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_theta = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def midpoint(*points: np.ndarray) -> np.ndarray:
    """Return the centroid of the given points."""
    return np.mean(points, axis=0)


def vertical_angle(v: np.ndarray) -> float:
    """Angle (degrees) between vector *v* and the downward vertical [0, 1] in 2-D."""
    down = np.array([0.0, 1.0])
    v2 = v[:2]
    norm = np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cos_theta = np.clip(np.dot(v2, down) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def horizontal_offset(a: np.ndarray, b: np.ndarray) -> float:
    """Signed horizontal difference a.x - b.x (positive = a is to the right of b)."""
    return float(a[0] - b[0])


def asymmetry_ratio(left: float, right: float) -> float:
    """Asymmetry ratio in [0, 1] where 0 is perfect symmetry.

    Uses the formula: |left - right| / max(left, right, epsilon)
    """
    denom = max(abs(left), abs(right), 1e-6)
    return abs(left - right) / denom
