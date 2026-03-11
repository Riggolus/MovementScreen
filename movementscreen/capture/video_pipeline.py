"""Video capture and frame pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from movementscreen.pose.estimator import PoseEstimator
from movementscreen.pose.landmarks import PoseFrame


@dataclass
class PipelineConfig:
    skip_frames: int = 0          # process every (skip_frames + 1)-th frame
    max_frames: int | None = None # cap total frames processed (None = all)
    draw_landmarks: bool = True   # overlay skeleton on preview frames
    show_preview: bool = False    # display a live OpenCV window


def iter_frames_from_file(
    video_path: str | Path,
    estimator: PoseEstimator,
    config: PipelineConfig | None = None,
) -> Iterator[tuple[PoseFrame, np.ndarray]]:
    """Yield (PoseFrame, annotated_bgr) tuples from a video file.

    Frames where no pose is detected are skipped.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python is not installed. Run: pip install opencv-python")

    config = config or PipelineConfig()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    processed = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % (config.skip_frames + 1) != 0:
                frame_idx += 1
                continue

            timestamp_ms = (frame_idx / fps) * 1000.0
            pose_frame = estimator.process_frame(bgr, frame_index=frame_idx, timestamp_ms=timestamp_ms)

            if pose_frame is not None:
                annotated = _annotate(bgr, pose_frame) if config.draw_landmarks else bgr.copy()
                if config.show_preview:
                    cv2.imshow("MovementScreen", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                yield pose_frame, annotated
                processed += 1

            frame_idx += 1
            if config.max_frames is not None and processed >= config.max_frames:
                break
    finally:
        cap.release()
        if config.show_preview:
            cv2.destroyAllWindows()


def iter_frames_from_webcam(
    estimator: PoseEstimator,
    camera_index: int = 0,
    config: PipelineConfig | None = None,
) -> Iterator[tuple[PoseFrame, np.ndarray]]:
    """Yield (PoseFrame, annotated_bgr) tuples from a webcam stream."""
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python is not installed.")

    config = config or PipelineConfig(show_preview=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    frame_idx = 0
    processed = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % (config.skip_frames + 1) != 0:
                frame_idx += 1
                continue

            pose_frame = estimator.process_frame(bgr, frame_index=frame_idx, timestamp_ms=frame_idx * 33.0)

            if pose_frame is not None:
                annotated = _annotate(bgr, pose_frame) if config.draw_landmarks else bgr.copy()
                if config.show_preview:
                    cv2.imshow("MovementScreen (q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                yield pose_frame, annotated
                processed += 1

            frame_idx += 1
            if config.max_frames is not None and processed >= config.max_frames:
                break
    finally:
        cap.release()
        if config.show_preview:
            cv2.destroyAllWindows()


def collect_frames(source: Iterator[tuple[PoseFrame, np.ndarray]]) -> list[PoseFrame]:
    """Drain an iterator and return all PoseFrames (discards annotated images)."""
    return [pf for pf, _ in source]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annotate(bgr: np.ndarray, frame: PoseFrame) -> np.ndarray:
    """Draw pose skeleton overlay on a copy of the frame."""
    try:
        import mediapipe as mp
        from movementscreen.pose.landmarks import LM
    except ImportError:
        return bgr.copy()

    annotated = bgr.copy()
    h, w = annotated.shape[:2]

    # Draw connections
    CONNECTIONS = [
        (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
        (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
        (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
        (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
        (LM.LEFT_HIP, LM.RIGHT_HIP),
        (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
        (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
        (LM.LEFT_ANKLE, LM.LEFT_HEEL), (LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX),
        (LM.RIGHT_ANKLE, LM.RIGHT_HEEL), (LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX),
    ]
    import cv2

    for a, b in CONNECTIONS:
        la, lb = frame.get(a), frame.get(b)
        if la.visible and lb.visible:
            pt_a = (int(la.x * w), int(la.y * h))
            pt_b = (int(lb.x * w), int(lb.y * h))
            cv2.line(annotated, pt_a, pt_b, (0, 255, 0), 2)

    # Draw landmark dots
    for lm in frame.landmarks:
        if lm.visible:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    return annotated
