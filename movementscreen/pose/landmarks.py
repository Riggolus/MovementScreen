"""MediaPipe landmark index definitions and named accessors."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class LM(IntEnum):
    """MediaPipe Pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class Landmark:
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    z: float  # depth relative to hips
    visibility: float  # [0, 1]

    def as_array(self, dims: int = 2) -> np.ndarray:
        if dims == 2:
            return np.array([self.x, self.y])
        return np.array([self.x, self.y, self.z])

    @property
    def visible(self) -> bool:
        return self.visibility > 0.5


@dataclass
class PoseFrame:
    """All landmarks for a single video frame."""
    landmarks: list[Landmark]
    frame_index: int
    timestamp_ms: float

    def get(self, lm: LM) -> Landmark:
        return self.landmarks[lm]

    def __getitem__(self, lm: LM) -> Landmark:
        return self.landmarks[lm]

    def bilateral_visible(self, left: LM, right: LM) -> bool:
        return self.get(left).visible and self.get(right).visible
