"""MovementScreen — CLI entry point.

Usage examples
--------------
Analyse a video file with the squat screen:
    python main.py --video path/to/squat.mp4 --screen squat

Analyse with the lunge screen (right lead):
    python main.py --video path/to/lunge.mp4 --screen lunge --lead-side right

Use webcam (press q to stop recording):
    python main.py --webcam --screen squat

Save a text report and plots:
    python main.py --video squat.mp4 --screen squat --output-dir ./results
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from movementscreen.capture.video_pipeline import (
    PipelineConfig,
    collect_frames,
    iter_frames_from_file,
    iter_frames_from_webcam,
)
from movementscreen.pose.estimator import PoseEstimator
from movementscreen.report.reporter import (
    plot_angle_ranges,
    plot_compensation_summary,
    print_report,
    save_text_report,
)
from movementscreen.screens.lunge import LungeScreen
from movementscreen.screens.overhead_reach import OverheadReachScreen
from movementscreen.screens.squat import SquatScreen


SCREENS = {
    "squat": lambda args: SquatScreen(),
    "lunge": lambda args: LungeScreen(lead_side=args.lead_side),
    "overhead": lambda args: OverheadReachScreen(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MovementScreen: computer-vision movement analysis tool"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to a video file")
    source.add_argument("--webcam", action="store_true", help="Use webcam as input")

    parser.add_argument(
        "--screen",
        choices=list(SCREENS),
        default="squat",
        help="Movement screen to run (default: squat)",
    )
    parser.add_argument(
        "--lead-side",
        choices=["left", "right"],
        default="left",
        help="Lead leg for lunge screen (default: left)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save reports and plots (optional)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (default: every other frame)",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Display live preview window during capture",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe model complexity 0=lite, 1=full, 2=heavy (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    screen = SCREENS[args.screen](args)
    config = PipelineConfig(
        skip_frames=args.skip_frames,
        show_preview=args.show_preview,
        draw_landmarks=True,
    )

    print(f"Running '{screen.name}' screen...")

    with PoseEstimator(model_complexity=args.model_complexity) as estimator:
        if args.webcam:
            source = iter_frames_from_webcam(estimator, config=config)
        else:
            if not args.video.exists():
                print(f"Error: video file not found: {args.video}", file=sys.stderr)
                sys.exit(1)
            source = iter_frames_from_file(args.video, estimator, config=config)

        frames = collect_frames(source)

    if not frames:
        print("No pose data detected. Check your video or camera input.")
        sys.exit(1)

    print(f"Collected {len(frames)} frames with valid pose data.")
    result = screen.run(frames)

    print_report(result)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = screen.name.lower().replace(" ", "_")

        text_path = args.output_dir / f"{stem}_report.txt"
        save_text_report(result, text_path)

        angles_plot = args.output_dir / f"{stem}_angles.png"
        plot_angle_ranges(result, angles_plot)

        if result.compensation_report.has_findings:
            comp_plot = args.output_dir / f"{stem}_compensations.png"
            plot_compensation_summary(result, comp_plot)


if __name__ == "__main__":
    main()
