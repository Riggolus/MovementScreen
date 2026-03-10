"""Pose estimation wrapper around MediaPipe Pose (Tasks API, mediapipe>=0.10.18)."""
from __future__ import annotations
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np

from movementscreen.pose.landmarks import LM, Landmark, PoseFrame

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

_MODELS_DIR = Path(__file__).parent / "models"

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def _get_model_path(model_complexity: int) -> Path:
    _MODELS_DIR.mkdir(exist_ok=True)
    model_path = _MODELS_DIR / _MODEL_NAMES[model_complexity]
    if not model_path.exists():
        url = _MODEL_URLS[model_complexity]
        print(f"Downloading pose model '{_MODEL_NAMES[model_complexity]}'...")
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")
    return model_path


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker (Tasks API) for single-frame and video-stream inference.

    Usage::

        with PoseEstimator() as estimator:
            frame = estimator.process_frame(bgr_image, frame_index=0, timestamp_ms=0.0)
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
        model_path = _get_model_path(model_complexity)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process_frame(
        self, bgr_frame: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0
    ) -> PoseFrame | None:
        """Run pose estimation on a single BGR frame.

        Returns a PoseFrame or None if no pose was detected.
        """
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not result.pose_landmarks:
            return None

        landmarks = [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )
            for lm in result.pose_landmarks[0]
        ]
        return PoseFrame(
            landmarks=landmarks,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()

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