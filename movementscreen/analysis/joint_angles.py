"""Compute named joint angles from a PoseFrame."""
from __future__ import annotations
from dataclasses import dataclass, field

from movementscreen.pose.landmarks import LM, PoseFrame
from movementscreen.utils.geometry import angle_between, midpoint


@dataclass
class JointAngles:
    """All relevant joint angles extracted from a single PoseFrame.

    Angles are in degrees. A value of None means the landmarks were not
    sufficiently visible to compute the angle reliably.
    """
    # Knee flexion (sagittal): hip-knee-ankle
    left_knee_flexion: float | None = None
    right_knee_flexion: float | None = None

    # Hip flexion (sagittal): shoulder-hip-knee
    left_hip_flexion: float | None = None
    right_hip_flexion: float | None = None

    # Ankle dorsiflexion (sagittal): knee-ankle-foot_index
    left_ankle_dorsiflexion: float | None = None
    right_ankle_dorsiflexion: float | None = None

    # Trunk lean: angle of shoulder-hip line from vertical
    trunk_lean_degrees: float | None = None

    # Lateral trunk shift: horizontal offset of mid-shoulder vs mid-hip (normalized)
    lateral_trunk_shift: float | None = None

    # Shoulder flexion (arms overhead)
    left_shoulder_flexion: float | None = None
    right_shoulder_flexion: float | None = None

    # Elbow angle
    left_elbow_angle: float | None = None
    right_elbow_angle: float | None = None

    # Knee valgus/varus: medial deviation of knee relative to hip-ankle line (2-D frontal)
    left_knee_frontal_angle: float | None = None
    right_knee_frontal_angle: float | None = None


def compute_joint_angles(frame: PoseFrame) -> JointAngles:
    """Extract all joint angles from *frame*."""
    angles = JointAngles()
    g = frame.get

    # --- Knee flexion ---
    if all(g(lm).visible for lm in (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)):
        angles.left_knee_flexion = angle_between(
            g(LM.LEFT_HIP).as_array(),
            g(LM.LEFT_KNEE).as_array(),
            g(LM.LEFT_ANKLE).as_array(),
        )
    if all(g(lm).visible for lm in (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)):
        angles.right_knee_flexion = angle_between(
            g(LM.RIGHT_HIP).as_array(),
            g(LM.RIGHT_KNEE).as_array(),
            g(LM.RIGHT_ANKLE).as_array(),
        )

    # --- Hip flexion ---
    if all(g(lm).visible for lm in (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE)):
        angles.left_hip_flexion = angle_between(
            g(LM.LEFT_SHOULDER).as_array(),
            g(LM.LEFT_HIP).as_array(),
            g(LM.LEFT_KNEE).as_array(),
        )
    if all(g(lm).visible for lm in (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE)):
        angles.right_hip_flexion = angle_between(
            g(LM.RIGHT_SHOULDER).as_array(),
            g(LM.RIGHT_HIP).as_array(),
            g(LM.RIGHT_KNEE).as_array(),
        )

    # --- Ankle dorsiflexion ---
    if all(g(lm).visible for lm in (LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX)):
        angles.left_ankle_dorsiflexion = angle_between(
            g(LM.LEFT_KNEE).as_array(),
            g(LM.LEFT_ANKLE).as_array(),
            g(LM.LEFT_FOOT_INDEX).as_array(),
        )
    if all(g(lm).visible for lm in (LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX)):
        angles.right_ankle_dorsiflexion = angle_between(
            g(LM.RIGHT_KNEE).as_array(),
            g(LM.RIGHT_ANKLE).as_array(),
            g(LM.RIGHT_FOOT_INDEX).as_array(),
        )

    # --- Trunk lean (angle of torso from vertical in image space) ---
    if all(g(lm).visible for lm in (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP)):
        import numpy as np
        from movementscreen.utils.geometry import vertical_angle

        mid_shoulder = midpoint(g(LM.LEFT_SHOULDER).as_array(), g(LM.RIGHT_SHOULDER).as_array())
        mid_hip = midpoint(g(LM.LEFT_HIP).as_array(), g(LM.RIGHT_HIP).as_array())
        trunk_vec = mid_shoulder - mid_hip  # pointing upward in image coords (y inverted)
        angles.trunk_lean_degrees = vertical_angle(-trunk_vec)  # negate because y is down

        # Lateral shift: horizontal distance between mid-shoulder and mid-hip
        angles.lateral_trunk_shift = float(mid_shoulder[0] - mid_hip[0])

    # --- Shoulder flexion ---
    if all(g(lm).visible for lm in (LM.LEFT_HIP, LM.LEFT_SHOULDER, LM.LEFT_ELBOW)):
        angles.left_shoulder_flexion = angle_between(
            g(LM.LEFT_HIP).as_array(),
            g(LM.LEFT_SHOULDER).as_array(),
            g(LM.LEFT_ELBOW).as_array(),
        )
    if all(g(lm).visible for lm in (LM.RIGHT_HIP, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW)):
        angles.right_shoulder_flexion = angle_between(
            g(LM.RIGHT_HIP).as_array(),
            g(LM.RIGHT_SHOULDER).as_array(),
            g(LM.RIGHT_ELBOW).as_array(),
        )

    # --- Elbow angle ---
    if all(g(lm).visible for lm in (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST)):
        angles.left_elbow_angle = angle_between(
            g(LM.LEFT_SHOULDER).as_array(),
            g(LM.LEFT_ELBOW).as_array(),
            g(LM.LEFT_WRIST).as_array(),
        )
    if all(g(lm).visible for lm in (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST)):
        angles.right_elbow_angle = angle_between(
            g(LM.RIGHT_SHOULDER).as_array(),
            g(LM.RIGHT_ELBOW).as_array(),
            g(LM.RIGHT_WRIST).as_array(),
        )

    # --- Knee frontal plane angle (valgus/varus proxy in 2-D) ---
    # Angle at knee between hip and ankle in the frontal (x-y) plane
    if all(g(lm).visible for lm in (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)):
        angles.left_knee_frontal_angle = angle_between(
            g(LM.LEFT_HIP).as_array(),
            g(LM.LEFT_KNEE).as_array(),
            g(LM.LEFT_ANKLE).as_array(),
        )
    if all(g(lm).visible for lm in (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)):
        angles.right_knee_frontal_angle = angle_between(
            g(LM.RIGHT_HIP).as_array(),
            g(LM.RIGHT_KNEE).as_array(),
            g(LM.RIGHT_ANKLE).as_array(),
        )

    return angles
