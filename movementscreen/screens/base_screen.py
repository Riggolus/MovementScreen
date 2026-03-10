"""Abstract base class for all movement screens."""
from __future__ import annotations
from abc import ABC, abstractmethod

from movementscreen.analysis.aggregator import TrialAggregator, TrialResult
from movementscreen.analysis.joint_angles import JointAngles, compute_joint_angles
from movementscreen.pose.landmarks import PoseFrame
from movementscreen.thresholds import ThresholdConfig


class BaseScreen(ABC):
    """A movement screen processes a sequence of PoseFrames and returns a TrialResult.

    Subclasses define the screen name and any screen-specific gate logic
    (e.g., only analyse frames where the subject is in the bottom position).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable screen name."""

    @property
    def screen_type(self) -> str:
        """Machine-readable screen identifier used for screen-specific threshold logic."""
        return ""

    def run(
        self,
        frames: list[PoseFrame],
        camera_angle: str = "anterior",
        thresholds: ThresholdConfig | None = None,
    ) -> TrialResult:
        """Process all frames and return the aggregated trial result."""
        aggregator = TrialAggregator(screen_name=self.name)
        for frame in frames:
            if self.accept_frame(frame, camera_angle):
                angles = compute_joint_angles(frame)
                angles = self.augment_angles(angles, frame)
                aggregator.add_frame(angles)
        return aggregator.finalize(camera_angle=camera_angle, thresholds=thresholds, screen_type=self.screen_type)

    def accept_frame(self, frame: PoseFrame, camera_angle: str = "anterior") -> bool:
        """Return True if this frame should be included in analysis.

        Override to gate on movement phase (e.g., bottom of squat only).
        Default: accept all frames.
        """
        return True

    def augment_angles(self, angles: JointAngles, frame: PoseFrame) -> JointAngles:
        """Optional hook to add screen-specific angle computations.

        Override to add or modify angles before they are stored.
        """
        return angles
