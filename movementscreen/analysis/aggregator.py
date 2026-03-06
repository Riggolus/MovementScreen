"""Aggregate per-frame angle data across a full movement screen trial."""
from __future__ import annotations
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Sequence

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.analysis.compensation import CompensationReport, Finding, Severity, detect_compensations


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
    ]

    def __init__(self, screen_name: str) -> None:
        self.screen_name = screen_name
        self._frames: list[JointAngles] = []

    def add_frame(self, angles: JointAngles) -> None:
        self._frames.append(angles)

    def finalize(self) -> TrialResult:
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

        # Build a representative "worst-case" JointAngles for compensation detection
        worst = JointAngles()
        for field_name in self._TRACKED_FIELDS:
            vals = [
                getattr(f, field_name)
                for f in self._frames
                if getattr(f, field_name) is not None
            ]
            if vals:
                # Use the value that deviates most from neutral (min for flexion)
                setattr(worst, field_name, min(vals))

        # Lateral trunk shift: use maximum deviation
        shifts = [f.lateral_trunk_shift for f in self._frames if f.lateral_trunk_shift is not None]
        if shifts:
            worst.lateral_trunk_shift = max(shifts, key=abs)

        compensation_report = detect_compensations(worst)

        return TrialResult(
            screen_name=self.screen_name,
            frame_count=len(self._frames),
            stats=stats,
            compensation_report=compensation_report,
        )
