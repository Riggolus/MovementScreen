"""Pose estimation wrapper around MediaPipe Pose."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Generator

import numpy as np

from movementscreen.pose.landmarks import LM, Landmark, PoseFrame

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


class PoseEstimator:
    """Wraps MediaPipe Pose for single-frame and video-stream inference.

    Usage::

        with PoseEstimator() as estimator:
            frame = estimator.process_frame(bgr_image, frame_index=0)
    """

    def __init__(
        self,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(
        self, bgr_frame: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0
    ) -> PoseFrame | None:
        """Run pose estimation on a single BGR frame.

        Returns a PoseFrame or None if no pose was detected.
        """
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )
            for lm in result.pose_landmarks.landmark
        ]
        return PoseFrame(
            landmarks=landmarks,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_) -> None:
        self.close()


@contextmanager
def pose_estimator(**kwargs) -> Generator[PoseEstimator, None, None]:
    """Context-manager convenience wrapper."""
    est = PoseEstimator(**kwargs)
    try:
        yield est
    finally:
        est.close()
