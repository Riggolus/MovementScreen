"""Compensation pattern detection from joint angle data."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.thresholds import ThresholdConfig
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


def detect_compensations(
    angles: JointAngles,
    camera_angle: str = "anterior",
    thresholds: Optional[ThresholdConfig] = None,
    screen_type: str = "",
) -> CompensationReport:
    """Evaluate a JointAngles snapshot and return all compensation findings.

    Args:
        angles:       Joint angle data for the worst-case frame of a trial.
        camera_angle: ``"anterior"`` or ``"lateral"`` — gates view-specific rules.
        thresholds:   Active threshold configuration.  Defaults to hardcoded
                      values when None (useful in tests and CLI usage).
    """
    t = thresholds or ThresholdConfig()
    report = CompensationReport()

    # --- Knee valgus (frontal plane collapse) ---
    for side, knee_angle in (
        ("Left", angles.left_knee_frontal_angle),
        ("Right", angles.right_knee_frontal_angle),
    ):
        if knee_angle is not None:
            sev = _severity_from_thresholds(
                knee_angle,
                mild=t.knee_valgus_moderate,
                moderate=t.knee_valgus_moderate,
                severe=t.knee_valgus_severe,
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

    # --- Excessive forward trunk lean (screen-type aware) ---
    if angles.trunk_lean_degrees is not None:
        if screen_type == "squat":
            # Optimal squat trunk lean is 20–40° depending on femur length.
            # Only flag genuinely excessive lean beyond that range.
            lean_mild     = t.squat_trunk_lean_mild
            lean_moderate = t.squat_trunk_lean_moderate
            lean_severe   = t.squat_trunk_lean_severe
            note = " (optimal squat range: 20–40°)"
        else:
            lean_mild     = t.trunk_lean_mild
            lean_moderate = t.trunk_lean_moderate
            lean_severe   = t.trunk_lean_severe
            note = ""
        sev = _severity_from_thresholds(
            angles.trunk_lean_degrees,
            mild=lean_mild,
            moderate=lean_moderate,
            severe=lean_severe,
        )
        if sev != Severity.NONE:
            report.add(Finding(
                name="Excessive Forward Trunk Lean",
                severity=sev,
                description=f"trunk is angled excessively forward from vertical{note}",
                metric_value=angles.trunk_lean_degrees,
                metric_label="trunk lean (deg)",
            ))

    # --- Lateral trunk shift ---
    if angles.lateral_trunk_shift is not None:
        shift = abs(angles.lateral_trunk_shift)
        sev = _severity_from_thresholds(
            shift,
            mild=t.lateral_shift_mild,
            moderate=t.lateral_shift_moderate,
            severe=t.lateral_shift_severe,
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
    for side, df in (
        ("Left", angles.left_ankle_dorsiflexion),
        ("Right", angles.right_ankle_dorsiflexion),
    ):
        if df is not None:
            sev = _severity_from_thresholds(
                df,
                mild=t.ankle_df_mild,
                moderate=t.ankle_df_moderate,
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

    # --- Pelvic tilt (anterior view) ---
    if camera_angle == "anterior" and angles.pelvic_tilt_degrees is not None:
        tilt = abs(angles.pelvic_tilt_degrees)
        sev = _severity_from_thresholds(
            tilt, t.pelvic_tilt_mild, t.pelvic_tilt_moderate, t.pelvic_tilt_severe
        )
        if sev != Severity.NONE:
            drop_side = "right" if angles.pelvic_tilt_degrees > 0 else "left"
            report.add(Finding(
                name=f"Pelvic Tilt ({drop_side} drop)",
                severity=sev,
                description=f"{drop_side.capitalize()} hip dropping lower than the opposite side — suggests hip weakness or leg-length difference",
                metric_value=round(tilt, 1),
                metric_label="pelvic tilt (deg)",
            ))

    # --- Lateral flexion (anterior view) ---
    if camera_angle == "anterior" and angles.lateral_flexion_degrees is not None:
        lflex = abs(angles.lateral_flexion_degrees)
        sev = _severity_from_thresholds(
            lflex, t.lateral_flexion_mild, t.lateral_flexion_moderate, t.lateral_flexion_severe
        )
        if sev != Severity.NONE and angles.lateral_trunk_shift is not None:
            direction = "right" if angles.lateral_trunk_shift > 0 else "left"
            report.add(Finding(
                name=f"Lateral Spinal Flexion ({direction})",
                severity=sev,
                description=f"trunk tilted laterally toward the {direction}",
                metric_value=lflex,
                metric_label="lateral flexion (deg)",
            ))

    # --- Spine segmental curvature (any view) ---
    if angles.spine_segmental_angle is not None:
        deviation = 180.0 - angles.spine_segmental_angle
        if deviation > 0:
            sev = _severity_from_thresholds(
                deviation, t.spine_curve_mild, t.spine_curve_moderate, t.spine_curve_severe
            )
            if sev != Severity.NONE:
                report.add(Finding(
                    name="Spinal Segmental Curvature",
                    severity=sev,
                    description=(
                        "segmental bend between thoracic and lumbar regions; "
                        "may indicate kyphosis or lordosis — confirm with lateral view"
                    ),
                    metric_value=round(deviation, 1),
                    metric_label="segmental deviation (deg)",
                ))

    # --- Lateral-view specific findings ---
    if camera_angle == "lateral":
        # Tibial angle — proxy for ankle dorsiflexion from lateral camera
        for side, angle in (
            ("Left",  angles.tibial_angle_left),
            ("Right", angles.tibial_angle_right),
        ):
            if angle is not None:
                sev = _severity_from_thresholds(
                    angle,
                    mild=t.tibial_angle_restricted_mild,
                    moderate=t.tibial_angle_restricted_mild,
                    severe=t.tibial_angle_restricted_severe,
                    lower_is_worse=True,
                )
                if sev != Severity.NONE:
                    report.add(Finding(
                        name=f"{side} Restricted Dorsiflexion",
                        severity=sev,
                        description=(
                            f"{side.lower()} tibia inclination suggests limited ankle dorsiflexion "
                            f"(optimal at squat depth: 30–40°)"
                        ),
                        metric_value=round(angle, 1),
                        metric_label="tibial angle (deg)",
                    ))
        if angles.head_forward_offset is not None:
            offset = abs(angles.head_forward_offset)
            sev = _severity_from_thresholds(
                offset, t.head_forward_mild, t.head_forward_moderate, t.head_forward_severe
            )
            if sev != Severity.NONE:
                report.add(Finding(
                    name="Head Forward Posture",
                    severity=sev,
                    description="ear positioned significantly ahead of shoulder in the sagittal plane",
                    metric_value=offset,
                    metric_label="head offset (normalized)",
                ))

        if angles.upper_trunk_angle is not None:
            sev = _severity_from_thresholds(
                angles.upper_trunk_angle,
                t.upper_trunk_mild, t.upper_trunk_moderate, t.upper_trunk_severe,
            )
            if sev != Severity.NONE:
                report.add(Finding(
                    name="Upper Trunk Flexion",
                    severity=sev,
                    description=(
                        "head and upper thoracic spine forward from vertical; "
                        "may indicate thoracic kyphosis or cervical hyperlordosis"
                    ),
                    metric_value=angles.upper_trunk_angle,
                    metric_label="upper trunk angle (deg)",
                ))

    # --- Bilateral symmetry checks ---
    _check_bilateral_asymmetry(
        report, t, "Knee Flexion", angles.left_knee_flexion, angles.right_knee_flexion
    )
    _check_bilateral_asymmetry(
        report, t, "Hip Flexion", angles.left_hip_flexion, angles.right_hip_flexion
    )
    _check_bilateral_asymmetry(
        report, t, "Shoulder Flexion", angles.left_shoulder_flexion, angles.right_shoulder_flexion
    )
    if camera_angle == "lateral":
        _check_bilateral_asymmetry(
            report, t, "Tibial Inclination", angles.tibial_angle_left, angles.tibial_angle_right
        )

    return report


def _check_bilateral_asymmetry(
    report: CompensationReport,
    t: ThresholdConfig,
    label: str,
    left: float | None,
    right: float | None,
) -> None:
    if left is None or right is None:
        return
    ratio = asymmetry_ratio(left, right)
    sev = _severity_from_thresholds(
        ratio,
        mild=t.asymmetry_mild,
        moderate=t.asymmetry_moderate,
        severe=t.asymmetry_severe,
    )
    if sev != Severity.NONE:
        report.add(Finding(
            name=f"Bilateral {label} Asymmetry",
            severity=sev,
            description=f"left ({left:.1f}°) vs right ({right:.1f}°) {label.lower()} differ significantly",
            metric_value=ratio,
            metric_label="asymmetry ratio",
        ))
