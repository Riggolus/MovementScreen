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

    Checks are gated by camera angle to match what is actually visible and
    clinically meaningful from each viewpoint:

    Frontal (anterior/posterior):
        Knee valgus, lateral trunk shift, pelvic tilt, lateral spinal
        flexion, bilateral L/R asymmetry.

    Sagittal (lateral):
        Forward trunk lean, ankle dorsiflexion (tibial angle + knee-ankle-foot
        proxy), head forward posture, upper trunk flexion, spinal segmental
        curvature, tibial bilateral asymmetry.
    """
    t = thresholds or ThresholdConfig()
    report = CompensationReport()

    is_frontal = camera_angle in ("anterior", "posterior")
    is_lateral = camera_angle == "lateral"

    # =========================================================
    # FRONTAL-PLANE CHECKS  (anterior / posterior camera only)
    # =========================================================
    if is_frontal:

        # --- Knee valgus (frontal plane collapse) ---
        # The hip-knee-ankle angle in 2D is a valid valgus proxy only when the
        # camera sees the frontal plane. From a lateral camera the same landmarks
        # produce the sagittal knee-flexion angle, which is meaningless here.
        for side, knee_angle in (
            ("Left",  angles.left_knee_frontal_angle),
            ("Right", angles.right_knee_frontal_angle),
        ):
            if knee_angle is not None:
                sev = _severity_from_thresholds(
                    knee_angle,
                    mild=t.knee_valgus_mild,
                    moderate=t.knee_valgus_moderate,
                    severe=t.knee_valgus_severe,
                    lower_is_worse=True,
                )
                if sev != Severity.NONE:
                    report.add(Finding(
                        name=f"{side} Knee Valgus",
                        severity=sev,
                        description=f"{side.lower()} knee collapsing medially",
                        metric_value=round(knee_angle, 1),
                        metric_label="knee frontal angle (deg)",
                    ))

        # --- Lateral trunk shift ---
        # Meaningful from frontal view: left/right shift of shoulders over hips.
        # From a lateral camera this axis becomes forward/backward, which is
        # a different (and already covered) measurement.
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
                    description=f"shoulders shifted laterally toward the {direction} relative to hips",
                    metric_value=round(shift, 3),
                    metric_label="shift (normalized)",
                ))

        # --- Pelvic tilt (hip line from horizontal) ---
        if angles.pelvic_tilt_degrees is not None:
            tilt = abs(angles.pelvic_tilt_degrees)
            sev = _severity_from_thresholds(
                tilt, t.pelvic_tilt_mild, t.pelvic_tilt_moderate, t.pelvic_tilt_severe
            )
            if sev != Severity.NONE:
                drop_side = "right" if angles.pelvic_tilt_degrees > 0 else "left"
                report.add(Finding(
                    name=f"Pelvic Tilt ({drop_side} drop)",
                    severity=sev,
                    description=(
                        f"{drop_side.capitalize()} hip dropping lower than the opposite side — "
                        "suggests hip abductor weakness or leg-length difference"
                    ),
                    metric_value=round(tilt, 1),
                    metric_label="pelvic tilt (deg)",
                ))

        # --- Lateral spinal flexion ---
        # arctan of horizontal/vertical trunk offset — only valid from frontal view.
        if angles.lateral_flexion_degrees is not None:
            lflex = abs(angles.lateral_flexion_degrees)
            sev = _severity_from_thresholds(
                lflex, t.lateral_flexion_mild, t.lateral_flexion_moderate, t.lateral_flexion_severe
            )
            if sev != Severity.NONE and angles.lateral_trunk_shift is not None:
                direction = "right" if angles.lateral_trunk_shift > 0 else "left"
                report.add(Finding(
                    name=f"Lateral Spinal Flexion ({direction})",
                    severity=sev,
                    description=f"trunk tilting laterally toward the {direction}",
                    metric_value=round(lflex, 1),
                    metric_label="lateral flexion (deg)",
                ))

        # --- Bilateral symmetry (L vs R) ---
        # Comparing left and right sides is only valid when both are visible
        # in the same plane — i.e. from a frontal camera.
        _check_bilateral_asymmetry(
            report, t, "Knee Flexion", angles.left_knee_flexion, angles.right_knee_flexion
        )
        _check_bilateral_asymmetry(
            report, t, "Hip Flexion", angles.left_hip_flexion, angles.right_hip_flexion
        )
        _check_bilateral_asymmetry(
            report, t, "Shoulder Flexion", angles.left_shoulder_flexion, angles.right_shoulder_flexion
        )

    # =========================================================
    # SAGITTAL-PLANE CHECKS  (lateral camera only)
    # =========================================================
    if is_lateral:

        # --- Excessive forward trunk lean ---
        # trunk_lean_degrees = angle of shoulder-hip line from vertical.
        # From a lateral camera this correctly captures forward lean in the
        # sagittal plane. From a frontal camera the same vector captures
        # lateral tilt (already covered by lateral_flexion_degrees).
        if angles.trunk_lean_degrees is not None:
            if screen_type == "squat":
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
                    description=f"trunk angled excessively forward from vertical{note}",
                    metric_value=round(angles.trunk_lean_degrees, 1),
                    metric_label="trunk lean (deg)",
                ))

        # --- Ankle dorsiflexion — tibial angle (primary lateral proxy) ---
        # Tibia angle from vertical at squat depth. Optimal: 30–40°.
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

        # --- Ankle dorsiflexion — knee-ankle-foot proxy (secondary) ---
        # Only flag if tibial angle didn't already fire for the same side,
        # to avoid duplicate findings.
        tibial_flagged = {f.name for f in report.findings if "Restricted Dorsiflexion" in f.name}
        for side, df in (
            ("Left",  angles.left_ankle_dorsiflexion),
            ("Right", angles.right_ankle_dorsiflexion),
        ):
            if df is not None and f"{side} Restricted Dorsiflexion" not in tibial_flagged:
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
                        metric_value=round(df, 1),
                        metric_label="ankle angle (deg)",
                    ))

        # --- Head forward posture ---
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
                    metric_value=round(offset, 3),
                    metric_label="head offset (normalized)",
                ))

        # --- Upper trunk flexion ---
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
                    metric_value=round(angles.upper_trunk_angle, 1),
                    metric_label="upper trunk angle (deg)",
                ))

        # --- Spinal segmental curvature ---
        # Ear-shoulder-hip angle: 180° = straight; deviations = curvature.
        # Only meaningful from lateral — from frontal it duplicates lateral flexion.
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
                            "segmental bend between upper and lower spine; "
                            "may indicate thoracic kyphosis or lumbar lordosis"
                        ),
                        metric_value=round(deviation, 1),
                        metric_label="segmental deviation (deg)",
                    ))

        # --- Tibial bilateral asymmetry ---
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
            metric_value=round(ratio, 3),
            metric_label="asymmetry ratio",
        ))
