"""Overhead reach / shoulder mobility screen."""
from __future__ import annotations

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.pose.landmarks import LM, PoseFrame
from movementscreen.screens.base_screen import BaseScreen
from movementscreen.utils.geometry import angle_between, midpoint


class OverheadReachScreen(BaseScreen):
    """Bilateral overhead reach screen.

    The subject raises both arms overhead. This screen evaluates:
    - Bilateral shoulder flexion range
    - Shoulder flexion asymmetry
    - Trunk extension compensation (arching lower back = trunk lean away from vertical)
    - Elbow extension
    """

    @property
    def name(self) -> str:
        return "Overhead Reach"

    @property
    def screen_type(self) -> str:
        return "overhead"

    def accept_frame(self, frame: PoseFrame, camera_angle: str = "anterior") -> bool:
        g = frame.get
        # Accept only frames where at least one wrist is above the nose
        for wrist, nose in ((LM.LEFT_WRIST, LM.NOSE), (LM.RIGHT_WRIST, LM.NOSE)):
            if g(wrist).visible and g(nose).visible:
                if g(wrist).y < g(nose).y:  # y is inverted: smaller = higher
                    return True
        return False

    def augment_angles(self, angles: JointAngles, frame: PoseFrame) -> JointAngles:
        g = frame.get
        # Wrist height relative to top of head (nose) as proxy for reach height
        if g(LM.LEFT_WRIST).visible and g(LM.NOSE).visible:
            angles.left_ankle_dorsiflexion = None  # not relevant here
        return angles
