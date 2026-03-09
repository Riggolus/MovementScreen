"""Compute named joint angles from a PoseFrame."""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from movementscreen.pose.landmarks import LM, PoseFrame
from movementscreen.utils.geometry import angle_between, midpoint, vertical_angle


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

    # Tibial angle: angle of tibia (knee→ankle) from vertical.
    # 0° = vertical tibia; 30–40° = optimal squat dorsiflexion.
    # Most meaningful from a lateral camera view.
    tibial_angle_left: float | None = None
    tibial_angle_right: float | None = None

    # Pelvic tilt: angle of the hip line from horizontal (anterior view).
    # Signed: positive = right hip drops lower; negative = left hip drops.
    pelvic_tilt_degrees: float | None = None

    # Lateral flexion: trunk tilt angle in the frontal plane (degrees from vertical)
    lateral_flexion_degrees: float | None = None

    # Upper trunk (cervicothoracic) angle from vertical: mid-ear → mid-shoulder segment
    upper_trunk_angle: float | None = None

    # Spine segmental angle: angle at mid-shoulder between ear–shoulder–hip segments.
    # 180° = straight spine; lower values indicate segmental curvature.
    spine_segmental_angle: float | None = None

    # Head forward offset: horizontal distance of mid-ear ahead of mid-shoulder
    # (normalized image coords). Meaningful from a lateral camera view.
    head_forward_offset: float | None = None


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

    # --- Trunk metrics (lean, lateral flexion, segmental curvature) ---
    if all(g(lm).visible for lm in (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP)):
        mid_shoulder = midpoint(g(LM.LEFT_SHOULDER).as_array(), g(LM.RIGHT_SHOULDER).as_array())
        mid_hip = midpoint(g(LM.LEFT_HIP).as_array(), g(LM.RIGHT_HIP).as_array())

        trunk_vec = mid_shoulder - mid_hip  # pointing upward in image coords (y inverted)
        angles.trunk_lean_degrees = vertical_angle(-trunk_vec)  # negate because y is down

        # Lateral shift: signed horizontal distance (shoulder vs hip)
        angles.lateral_trunk_shift = float(mid_shoulder[0] - mid_hip[0])

        # Lateral flexion angle: convert horizontal offset to degrees
        vert_dist = float(mid_hip[1] - mid_shoulder[1])  # positive (hip is below shoulder)
        if vert_dist > 0:
            angles.lateral_flexion_degrees = float(
                np.degrees(np.arctan2(abs(angles.lateral_trunk_shift), vert_dist))
            )

        # --- Spine segmentation using ears ---
        if frame.bilateral_visible(LM.LEFT_EAR, LM.RIGHT_EAR):
            mid_ear = midpoint(g(LM.LEFT_EAR).as_array(), g(LM.RIGHT_EAR).as_array())

            # Upper trunk angle: ear→shoulder segment deviation from vertical
            upper_vec = mid_shoulder - mid_ear  # pointing downward (ear to shoulder)
            angles.upper_trunk_angle = vertical_angle(upper_vec)

            # Spine segmental angle: angle at mid-shoulder between ear–shoulder–hip
            # 180° = straight; deviations indicate curvature between segments
            angles.spine_segmental_angle = angle_between(mid_ear, mid_shoulder, mid_hip)

            # Head forward offset: horizontal ear position relative to shoulder
            # Positive = ear is ahead/to-the-right of shoulder (camera dependent)
            angles.head_forward_offset = float(mid_ear[0] - mid_shoulder[0])

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

    # --- Tibial angle (tibia from vertical) ---
    # Measures forward inclination of the shin — proxy for ankle dorsiflexion from lateral view.
    # Optimal squat depth: 30–40°. Values below ~25° suggest restricted dorsiflexion.
    if all(g(lm).visible for lm in (LM.LEFT_KNEE, LM.LEFT_ANKLE)):
        tibia_left = g(LM.LEFT_ANKLE).as_array() - g(LM.LEFT_KNEE).as_array()
        angles.tibial_angle_left = vertical_angle(tibia_left)
    if all(g(lm).visible for lm in (LM.RIGHT_KNEE, LM.RIGHT_ANKLE)):
        tibia_right = g(LM.RIGHT_ANKLE).as_array() - g(LM.RIGHT_KNEE).as_array()
        angles.tibial_angle_right = vertical_angle(tibia_right)

    # --- Pelvic tilt (hip line from horizontal, anterior view) ---
    # Signed: positive = right hip lower; negative = left hip lower.
    if frame.bilateral_visible(LM.LEFT_HIP, LM.RIGHT_HIP):
        left_hip  = g(LM.LEFT_HIP).as_array()
        right_hip = g(LM.RIGHT_HIP).as_array()
        horiz = float(abs(right_hip[0] - left_hip[0]))
        if horiz > 0.01:  # guard against coincident points
            vert = float(right_hip[1] - left_hip[1])  # + = right hip lower (y down)
            angles.pelvic_tilt_degrees = float(np.degrees(np.arctan2(vert, horiz)))

    return angles
