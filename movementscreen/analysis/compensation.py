"""Compensation pattern detection from joint angle data."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.utils.geometry import asymmetry_ratio


class Severity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class Finding:
    """A single detected compensation pattern."""
    name: str
    severity: Severity
    description: str
    metric_value: float | None = None
    metric_label: str | None = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.name}: {self.description}"]
        if self.metric_value is not None and self.metric_label is not None:
            parts.append(f"  ({self.metric_label}: {self.metric_value:.1f})")
        return "\n".join(parts)


@dataclass
class CompensationReport:
    """All compensation findings for a movement screen or a single frame."""
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def worst_severity(self) -> Severity:
        order = [Severity.NONE, Severity.MILD, Severity.MODERATE, Severity.SEVERE]
        if not self.findings:
            return Severity.NONE
        return max(self.findings, key=lambda f: order.index(f.severity)).severity

    def __str__(self) -> str:
        if not self.findings:
            return "No compensations detected."
        return "\n".join(str(f) for f in self.findings)


# ---------------------------------------------------------------------------
# Thresholds (degrees unless noted)
# ---------------------------------------------------------------------------

# Knee valgus: angle at knee (frontal) < threshold suggests medial collapse
KNEE_VALGUS_MODERATE = 170.0   # angle in degrees (straighter = more valgus)
KNEE_VALGUS_SEVERE = 160.0

# Excessive forward trunk lean
TRUNK_LEAN_MILD = 20.0
TRUNK_LEAN_MODERATE = 35.0
TRUNK_LEAN_SEVERE = 50.0

# Heel rise proxy: low ankle dorsiflexion angle during squat
ANKLE_DF_MILD = 100.0       # angle hip-knee-ankle-foot less than threshold
ANKLE_DF_MODERATE = 90.0

# Lateral trunk shift (normalized image coords)
LATERAL_SHIFT_MILD = 0.02
LATERAL_SHIFT_MODERATE = 0.05
LATERAL_SHIFT_SEVERE = 0.08

# Bilateral asymmetry ratio
ASYMMETRY_MILD = 0.10
ASYMMETRY_MODERATE = 0.20
ASYMMETRY_SEVERE = 0.35


def _severity_from_thresholds(
    value: float,
    mild: float,
    moderate: float,
    severe: float | None = None,
    lower_is_worse: bool = False,
) -> Severity:
    if lower_is_worse:
        if severe is not None and value <= severe:
            return Severity.SEVERE
        if value <= moderate:
            return Severity.MODERATE
        if value <= mild:
            return Severity.MILD
        return Severity.NONE
    else:
        if severe is not None and value >= severe:
            return Severity.SEVERE
        if value >= moderate:
            return Severity.MODERATE
        if value >= mild:
            return Severity.MILD
        return Severity.NONE


def detect_compensations(angles: JointAngles) -> CompensationReport:
    """Evaluate a JointAngles snapshot and return all compensation findings."""
    report = CompensationReport()

    # --- Knee valgus (frontal plane collapse) ---
    for side, knee_angle in (("Left", angles.left_knee_frontal_angle), ("Right", angles.right_knee_frontal_angle)):
        if knee_angle is not None:
            sev = _severity_from_thresholds(
                knee_angle,
                mild=KNEE_VALGUS_MODERATE,
                moderate=KNEE_VALGUS_MODERATE,
                severe=KNEE_VALGUS_SEVERE,
                lower_is_worse=True,
            )
            if sev != Severity.NONE:
                report.add(Finding(
                    name=f"{side} Knee Valgus",
                    severity=sev,
                    description=f"{side.lower()} knee collapsing medially",
                    metric_value=knee_angle,
                    metric_label="knee frontal angle (deg)",
                ))

    # --- Excessive forward trunk lean ---
    if angles.trunk_lean_degrees is not None:
        sev = _severity_from_thresholds(
            angles.trunk_lean_degrees,
            mild=TRUNK_LEAN_MILD,
            moderate=TRUNK_LEAN_MODERATE,
            severe=TRUNK_LEAN_SEVERE,
        )
        if sev != Severity.NONE:
            report.add(Finding(
                name="Excessive Forward Trunk Lean",
                severity=sev,
                description="trunk is angled excessively forward from vertical",
                metric_value=angles.trunk_lean_degrees,
                metric_label="trunk lean (deg)",
            ))

    # --- Lateral trunk shift ---
    if angles.lateral_trunk_shift is not None:
        shift = abs(angles.lateral_trunk_shift)
        sev = _severity_from_thresholds(
            shift,
            mild=LATERAL_SHIFT_MILD,
            moderate=LATERAL_SHIFT_MODERATE,
            severe=LATERAL_SHIFT_SEVERE,
        )
        if sev != Severity.NONE:
            direction = "right" if angles.lateral_trunk_shift > 0 else "left"
            report.add(Finding(
                name=f"Lateral Trunk Shift ({direction})",
                severity=sev,
                description=f"shoulders shifted laterally relative to hips toward {direction}",
                metric_value=shift,
                metric_label="shift (normalized)",
            ))

    # --- Reduced ankle dorsiflexion (heel rise proxy) ---
    for side, df in (("Left", angles.left_ankle_dorsiflexion), ("Right", angles.right_ankle_dorsiflexion)):
        if df is not None:
            sev = _severity_from_thresholds(
                df,
                mild=ANKLE_DF_MILD,
                moderate=ANKLE_DF_MODERATE,
                severe=None,
                lower_is_worse=True,
            )
            if sev != Severity.NONE:
                report.add(Finding(
                    name=f"{side} Heel Rise / Limited Dorsiflexion",
                    severity=sev,
                    description=f"restricted ankle dorsiflexion on {side.lower()} side",
                    metric_value=df,
                    metric_label="ankle angle (deg)",
                ))

    # --- Bilateral symmetry checks ---
    _check_bilateral_asymmetry(report, "Knee Flexion", angles.left_knee_flexion, angles.right_knee_flexion)
    _check_bilateral_asymmetry(report, "Hip Flexion", angles.left_hip_flexion, angles.right_hip_flexion)
    _check_bilateral_asymmetry(report, "Shoulder Flexion", angles.left_shoulder_flexion, angles.right_shoulder_flexion)

    return report


def _check_bilateral_asymmetry(
    report: CompensationReport,
    label: str,
    left: float | None,
    right: float | None,
) -> None:
    if left is None or right is None:
        return
    ratio = asymmetry_ratio(left, right)
    sev = _severity_from_thresholds(
        ratio,
        mild=ASYMMETRY_MILD,
        moderate=ASYMMETRY_MODERATE,
        severe=ASYMMETRY_SEVERE,
    )
    if sev != Severity.NONE:
        report.add(Finding(
            name=f"Bilateral {label} Asymmetry",
            severity=sev,
            description=f"left ({left:.1f}°) vs right ({right:.1f}°) {label.lower()} differ significantly",
            metric_value=ratio,
            metric_label="asymmetry ratio",
        ))
