"""Tests for joint angle computation from synthetic PoseFrames."""
import numpy as np
import pytest

from movementscreen.pose.landmarks import LM, Landmark, PoseFrame
from movementscreen.analysis.joint_angles import compute_joint_angles


def _landmark(x: float, y: float, z: float = 0.0, vis: float = 1.0) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=vis)


def _make_frame(positions: dict[LM, tuple[float, float]]) -> PoseFrame:
    """Build a PoseFrame from a dict of {LM: (x, y)} with all other landmarks at (0,0) invisible."""
    landmarks = [_landmark(0.0, 0.0, vis=0.0)] * 33
    for lm, (x, y) in positions.items():
        landmarks[lm] = _landmark(x, y, vis=1.0)
    return PoseFrame(landmarks=landmarks, frame_index=0, timestamp_ms=0.0)


def test_knee_flexion_90_degrees():
    # Arrange a right-angle knee: hip directly above knee, ankle directly to the right
    frame = _make_frame({
        LM.LEFT_HIP: (0.5, 0.3),
        LM.LEFT_KNEE: (0.5, 0.6),
        LM.LEFT_ANKLE: (0.8, 0.6),
    })
    angles = compute_joint_angles(frame)
    assert angles.left_knee_flexion is not None
    assert abs(angles.left_knee_flexion - 90.0) < 1.0


def test_knee_flexion_straight_leg():
    # Straight leg: hip, knee, ankle collinear vertically → ~180°
    frame = _make_frame({
        LM.LEFT_HIP: (0.5, 0.2),
        LM.LEFT_KNEE: (0.5, 0.5),
        LM.LEFT_ANKLE: (0.5, 0.8),
    })
    angles = compute_joint_angles(frame)
    assert angles.left_knee_flexion is not None
    assert abs(angles.left_knee_flexion - 180.0) < 1.0


def test_invisible_landmarks_return_none():
    # No landmarks visible → all angles should be None
    landmarks = [_landmark(0.0, 0.0, vis=0.0)] * 33
    frame = PoseFrame(landmarks=landmarks, frame_index=0, timestamp_ms=0.0)
    angles = compute_joint_angles(frame)
    assert angles.left_knee_flexion is None
    assert angles.right_knee_flexion is None
    assert angles.trunk_lean_degrees is None


def test_trunk_lean_computed_when_landmarks_visible():
    frame = _make_frame({
        LM.LEFT_SHOULDER: (0.4, 0.2),
        LM.RIGHT_SHOULDER: (0.6, 0.2),
        LM.LEFT_HIP: (0.4, 0.6),
        LM.RIGHT_HIP: (0.6, 0.6),
    })
    angles = compute_joint_angles(frame)
    assert angles.trunk_lean_degrees is not None
    assert angles.lateral_trunk_shift is not None
