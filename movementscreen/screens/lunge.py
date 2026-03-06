"""Forward lunge movement screen."""
from __future__ import annotations

from movementscreen.pose.landmarks import LM, PoseFrame
from movementscreen.screens.base_screen import BaseScreen
from movementscreen.utils.geometry import angle_between

LUNGE_DEPTH_THRESHOLD_DEGREES = 120.0  # lead knee angle < 120° → near bottom


class LungeScreen(BaseScreen):
    """Forward lunge movement screen.

    Detects:
    - Lead knee valgus
    - Trunk lean and lateral shift
    - Rear knee drop (hip flexion)
    - Bilateral asymmetry when compared between left and right lead legs
    """

    def __init__(self, lead_side: str = "left", at_depth_only: bool = True) -> None:
        if lead_side not in ("left", "right"):
            raise ValueError("lead_side must be 'left' or 'right'")
        self._lead_side = lead_side
        self._at_depth_only = at_depth_only

    @property
    def name(self) -> str:
        return f"Forward Lunge ({self._lead_side.capitalize()} Lead)"

    def accept_frame(self, frame: PoseFrame) -> bool:
        if not self._at_depth_only:
            return True
        g = frame.get
        if self._lead_side == "left":
            hip, knee, ankle = LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE
        else:
            hip, knee, ankle = LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE

        if all(g(lm).visible for lm in (hip, knee, ankle)):
            knee_angle = angle_between(
                g(hip).as_array(), g(knee).as_array(), g(ankle).as_array()
            )
            return knee_angle < LUNGE_DEPTH_THRESHOLD_DEGREES
        return False
