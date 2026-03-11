"""Overhead squat / bodyweight squat screen."""
from __future__ import annotations

from movementscreen.pose.landmarks import LM, PoseFrame
from movementscreen.screens.base_screen import BaseScreen
from movementscreen.utils.geometry import angle_between

# Sagittal depth gate (lateral camera only): knee flexion angle < this → at depth.
# From lateral view the 2D hip-knee-ankle angle captures sagittal flexion correctly.
SQUAT_DEPTH_THRESHOLD_DEGREES = 115.0  # knee angle < 115° → at depth

# Frontal depth gate (anterior/posterior camera): normalised hip height relative to knee.
# When hip_norm >= this fraction of knee_norm the hips are near/at/below knee level.
# 0.85 = hips within ~85 % of the knee's normalised height → roughly parallel or below.
FRONTAL_DEPTH_HIP_FRACTION = 0.85


class SquatScreen(BaseScreen):
    """Bilateral bodyweight squat movement screen.

    Detects:
    - Knee valgus
    - Excessive forward trunk lean
    - Heel rise / limited ankle dorsiflexion
    - Lateral trunk shift
    - Bilateral knee and hip asymmetry
    """

    def __init__(self, at_depth_only: bool = True) -> None:
        self._at_depth_only = at_depth_only

    @property
    def name(self) -> str:
        return "Bodyweight Squat"

    @property
    def screen_type(self) -> str:
        return "squat"

    def accept_frame(self, frame: PoseFrame, camera_angle: str = "anterior") -> bool:
        if not self._at_depth_only:
            return True
        g = frame.get

        if camera_angle in ("anterior", "posterior"):
            # From a frontal camera the 2D knee angle does NOT capture sagittal depth
            # (the knee moves forward in Z, which is discarded).  Instead, use the
            # vertical proximity of hip to knee as a depth proxy, normalised by
            # body height so the check is frame-scale independent.
            shoulder_ys = [g(lm).as_array()[1] for lm in (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
                           if g(lm).visible]
            ankle_ys    = [g(lm).as_array()[1] for lm in (LM.LEFT_ANKLE,    LM.RIGHT_ANKLE)
                           if g(lm).visible]
            if not shoulder_ys or not ankle_ys:
                return False
            top_y    = min(shoulder_ys)
            bottom_y = max(ankle_ys)
            body_h   = bottom_y - top_y
            if body_h < 0.01:
                return False

            for hip_lm, knee_lm in (
                (LM.LEFT_HIP,  LM.LEFT_KNEE),
                (LM.RIGHT_HIP, LM.RIGHT_KNEE),
            ):
                if g(hip_lm).visible and g(knee_lm).visible:
                    hip_norm  = (g(hip_lm).as_array()[1]  - top_y) / body_h
                    knee_norm = (g(knee_lm).as_array()[1] - top_y) / body_h
                    if knee_norm > 0 and hip_norm >= knee_norm * FRONTAL_DEPTH_HIP_FRACTION:
                        return True
            return False

        else:
            # Lateral camera: 2D knee flexion correctly captures sagittal depth.
            for hip, knee, ankle in (
                (LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE),
                (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
            ):
                if all(g(lm).visible for lm in (hip, knee, ankle)):
                    knee_angle = angle_between(
                        g(hip).as_array(), g(knee).as_array(), g(ankle).as_array()
                    )
                    if knee_angle < SQUAT_DEPTH_THRESHOLD_DEGREES:
                        return True
            return False
