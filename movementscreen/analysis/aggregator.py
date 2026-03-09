"""Aggregate per-frame angle data across a full movement screen trial."""
from __future__ import annotations
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Sequence

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.analysis.compensation import CompensationReport, Finding, Severity, detect_compensations
from movementscreen.thresholds import ThresholdConfig


@dataclass
class TrialStats:
    """Descriptive statistics for a single angle metric across frames."""
    name: str
    values: list[float] = field(default_factory=list)

    @property
    def min(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def max(self) -> float | None:
        return max(self.values) if self.values else None

    @property
    def mean(self) -> float | None:
        return mean(self.values) if self.values else None

    @property
    def sd(self) -> float | None:
        return stdev(self.values) if len(self.values) > 1 else None


@dataclass
class TrialResult:
    """Full analysis result for a single movement screen trial."""
    screen_name: str
    frame_count: int
    stats: dict[str, TrialStats] = field(default_factory=dict)
    compensation_report: CompensationReport = field(default_factory=CompensationReport)

    def summary(self) -> str:
        lines = [
            f"=== {self.screen_name} Trial Summary ===",
            f"Frames analyzed: {self.frame_count}",
            "",
            "--- Joint Angle Ranges ---",
        ]
        for stat in self.stats.values():
            if stat.min is not None:
                lines.append(
                    f"  {stat.name}: min={stat.min:.1f}°  max={stat.max:.1f}°  "
                    f"mean={stat.mean:.1f}°"
                )
        lines += ["", "--- Compensations ---", str(self.compensation_report)]
        return "\n".join(lines)


class TrialAggregator:
    """Collect per-frame JointAngles and produce a TrialResult."""

    _TRACKED_FIELDS = [
        "left_knee_flexion", "right_knee_flexion",
        "left_hip_flexion", "right_hip_flexion",
        "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
        "trunk_lean_degrees",
        "left_shoulder_flexion", "right_shoulder_flexion",
        "lateral_flexion_degrees",
        "upper_trunk_angle",
        "spine_segmental_angle",
        "tibial_angle_left", "tibial_angle_right",  # lower = worse (restricted DF)
        "pelvic_tilt_degrees",                      # handled separately (signed)
    ]

    def __init__(self, screen_name: str) -> None:
        self.screen_name = screen_name
        self._frames: list[JointAngles] = []

    def add_frame(self, angles: JointAngles) -> None:
        self._frames.append(angles)

    def finalize(self, camera_angle: str = "anterior", thresholds: ThresholdConfig | None = None, screen_type: str = "") -> TrialResult:
        stats: dict[str, TrialStats] = {}
        for field_name in self._TRACKED_FIELDS:
            label = field_name.replace("_", " ").title()
            ts = TrialStats(name=label)
            for frame in self._frames:
                v = getattr(frame, field_name)
                if v is not None:
                    ts.values.append(v)
            if ts.values:
                stats[field_name] = ts

        # Build a representative JointAngles for compensation detection.
        # Strategy: use the median of each field so that a single outlier frame
        # (e.g. a transition frame at the depth gate boundary, or a MediaPipe
        # tracking glitch) does not drive the entire result.  For directional
        # fields that must reflect the worst bias we still use the appropriate
        # extreme, but only after clamping outliers via the sorted middle value.
        worst = JointAngles()

        # Fields where LOWER is worse (more flexion, more restriction, more curvature)
        _min_is_worse = {
            "left_knee_flexion", "right_knee_flexion",
            "left_hip_flexion", "right_hip_flexion",
            "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
            "tibial_angle_left", "tibial_angle_right",
            "spine_segmental_angle",
        }
        # Fields where HIGHER is worse (more lean, more lateral flex)
        _max_is_worse = {
            "trunk_lean_degrees",
            "lateral_flexion_degrees",
            "upper_trunk_angle",
        }

        for field_name in self._TRACKED_FIELDS:
            vals = sorted(
                (getattr(f, field_name) for f in self._frames if getattr(f, field_name) is not None)
            )
            if not vals:
                continue
            n = len(vals)
            if field_name in _min_is_worse:
                # Use the 25th-percentile value — worse than median but not a single-frame outlier
                idx = max(0, int(n * 0.25))
                setattr(worst, field_name, vals[idx])
            elif field_name in _max_is_worse:
                # Use the 75th-percentile value
                idx = min(n - 1, int(n * 0.75))
                setattr(worst, field_name, vals[idx])
            else:
                setattr(worst, field_name, median(vals))

        # Lateral trunk shift: use 75th-percentile absolute deviation
        shifts = sorted(
            (f.lateral_trunk_shift for f in self._frames if f.lateral_trunk_shift is not None),
            key=abs,
        )
        if shifts:
            idx = min(len(shifts) - 1, int(len(shifts) * 0.75))
            worst.lateral_trunk_shift = shifts[idx]

        # Pelvic tilt: signed; use 75th-percentile absolute deviation
        pelvic_vals = sorted(
            (f.pelvic_tilt_degrees for f in self._frames if f.pelvic_tilt_degrees is not None),
            key=abs,
        )
        if pelvic_vals:
            idx = min(len(pelvic_vals) - 1, int(len(pelvic_vals) * 0.75))
            worst.pelvic_tilt_degrees = pelvic_vals[idx]

        # Head forward offset: 75th-percentile absolute deviation
        head_offsets = sorted(
            (f.head_forward_offset for f in self._frames if f.head_forward_offset is not None),
            key=abs,
        )
        if head_offsets:
            idx = min(len(head_offsets) - 1, int(len(head_offsets) * 0.75))
            worst.head_forward_offset = head_offsets[idx]

        compensation_report = detect_compensations(worst, camera_angle=camera_angle, thresholds=thresholds, screen_type=screen_type)

        return TrialResult(
            screen_name=self.screen_name,
            frame_count=len(self._frames),
            stats=stats,
            compensation_report=compensation_report,
        )
