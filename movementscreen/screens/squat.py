"""Overhead squat / bodyweight squat screen."""
from __future__ import annotations

from movementscreen.pose.landmarks import LM, PoseFrame
from movementscreen.screens.base_screen import BaseScreen

# Gate: only analyse frames where the knee flexion exceeds this angle
# (i.e., the subject is at or near the bottom of the squat).
SQUAT_DEPTH_THRESHOLD_DEGREES = 130.0  # knee angle < 130° → at depth


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

    def accept_frame(self, frame: PoseFrame) -> bool:
        if not self._at_depth_only:
            return True
        g = frame.get
        # Accept only frames near squat depth on at least one side
        from movementscreen.utils.geometry import angle_between
        for hip, knee, ankle in (
            (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
            (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
        ):
            if all(g(lm).visible for lm in (hip, knee, ankle)):
                knee_angle = angle_between(
                    g(hip).as_array(), g(knee).as_array(), g(ankle).as_array()
                )
                if knee_angle < SQUAT_DEPTH_THRESHOLD_DEGREES:
                    return True
        return False
