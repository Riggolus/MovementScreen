"""Compensation pattern detection from joint angle data."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.thresholds import ThresholdConfig
from movementscreen.utils.geometry import asymmetry_ratio


class Grade(Enum):
    """A–F compensation grade.

    A  No compensation — movement within normal limits
    B  Minimal — barely detectable deviation, monitor only
    C  Mild — noticeable compensation, address in programming
    D  Moderate — clear dysfunction, prioritise in plan
    E  Significant — marked compensation, intervention needed
    F  Severe — major dysfunction, refer or restrict loading
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


# Backward-compatible alias so any code still importing Severity doesn't break immediately
Severity = Grade


@dataclass
class Finding:
    """A single detected compensation pattern."""
    name: str
    severity: Grade
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
    def worst_severity(self) -> Grade:
        order = [Grade.A, Grade.B, Grade.C, Grade.D, Grade.E, Grade.F]
        if not self.findings:
            return Grade.A
        return max(self.findings, key=lambda f: order.index(f.severity)).severity

    def __str__(self) -> str:
        if not self.findings:
            return "No compensations detected."
        return "\n".join(str(f) for f in self.findings)


def _grade_from_thresholds(
    value: float,
    b: float,
    c: float,
    d: float,
    e: float | None = None,
    f: float | None = None,
    lower_is_worse: bool = False,
) -> Grade:
    """Return the compensation Grade for *value* given ordered band boundaries.

    For higher-is-worse metrics (e.g. trunk lean):
        A < b ≤ B < c ≤ C < d ≤ D < e ≤ E < f ≤ F

    For lower-is-worse metrics (e.g. knee frontal angle):
        A > b ≥ B > c ≥ C > d ≥ D > e ≥ E > f ≥ F
    """
    if lower_is_worse:
        if f is not None and value <= f:
            return Grade.F
        if e is not None and value <= e:
            return Grade.E
        if value <= d:
            return Grade.D
        if value <= c:
            return Grade.C
        if value <= b:
            return Grade.B
        return Grade.A
    else:
        if f is not None and value >= f:
            return Grade.F
        if e is not None and value >= e:
            return Grade.E
        if value >= d:
            return Grade.D
        if value >= c:
            return Grade.C
        if value >= b:
            return Grade.B
        return Grade.A


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
                sev = _grade_from_thresholds(
                    knee_angle,
                    b=t.knee_valgus_b, c=t.knee_valgus_c, d=t.knee_valgus_d,
                    e=t.knee_valgus_e, f=t.knee_valgus_f,
                    lower_is_worse=True,
                )
                if sev != Grade.A:
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
            sev = _grade_from_thresholds(
                shift,
                b=t.lateral_shift_b, c=t.lateral_shift_c, d=t.lateral_shift_d,
                e=t.lateral_shift_e, f=t.lateral_shift_f,
            )
            if sev != Grade.A:
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
            sev = _grade_from_thresholds(
                tilt,
                b=t.pelvic_tilt_b, c=t.pelvic_tilt_c, d=t.pelvic_tilt_d,
                e=t.pelvic_tilt_e, f=t.pelvic_tilt_f,
            )
            if sev != Grade.A:
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
            sev = _grade_from_thresholds(
                lflex,
                b=t.lateral_flexion_b, c=t.lateral_flexion_c, d=t.lateral_flexion_d,
                e=t.lateral_flexion_e, f=t.lateral_flexion_f,
            )
            if sev != Grade.A and angles.lateral_trunk_shift is not None:
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
                lb, lc, ld = t.squat_trunk_lean_b, t.squat_trunk_lean_c, t.squat_trunk_lean_d
                le, lf = t.squat_trunk_lean_e, t.squat_trunk_lean_f
                note = " (optimal squat range: 20–40°)"
            else:
                lb, lc, ld = t.trunk_lean_b, t.trunk_lean_c, t.trunk_lean_d
                le, lf = t.trunk_lean_e, t.trunk_lean_f
                note = ""
            sev = _grade_from_thresholds(
                angles.trunk_lean_degrees,
                b=lb, c=lc, d=ld, e=le, f=lf,
            )
            if sev != Grade.A:
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
                sev = _grade_from_thresholds(
                    angle,
                    b=t.tibial_angle_b, c=t.tibial_angle_c, d=t.tibial_angle_d,
                    e=t.tibial_angle_e, f=t.tibial_angle_f,
                    lower_is_worse=True,
                )
                if sev != Grade.A:
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
                sev = _grade_from_thresholds(
                    df,
                    b=t.ankle_df_b, c=t.ankle_df_c, d=t.ankle_df_d,
                    e=t.ankle_df_e, f=t.ankle_df_f,
                    lower_is_worse=True,
                )
                if sev != Grade.A:
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
            sev = _grade_from_thresholds(
                offset,
                b=t.head_forward_b, c=t.head_forward_c, d=t.head_forward_d,
                e=t.head_forward_e, f=t.head_forward_f,
            )
            if sev != Grade.A:
                report.add(Finding(
                    name="Head Forward Posture",
                    severity=sev,
                    description="ear positioned significantly ahead of shoulder in the sagittal plane",
                    metric_value=round(offset, 3),
                    metric_label="head offset (normalized)",
                ))

        # --- Upper trunk flexion ---
        if angles.upper_trunk_angle is not None:
            sev = _grade_from_thresholds(
                angles.upper_trunk_angle,
                b=t.upper_trunk_b, c=t.upper_trunk_c, d=t.upper_trunk_d,
                e=t.upper_trunk_e, f=t.upper_trunk_f,
            )
            if sev != Grade.A:
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
                sev = _grade_from_thresholds(
                    deviation,
                    b=t.spine_curve_b, c=t.spine_curve_c, d=t.spine_curve_d,
                    e=t.spine_curve_e, f=t.spine_curve_f,
                )
                if sev != Grade.A:
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
    sev = _grade_from_thresholds(
        ratio,
        b=t.asymmetry_b, c=t.asymmetry_c, d=t.asymmetry_d,
        e=t.asymmetry_e, f=t.asymmetry_f,
    )
    if sev != Grade.A:
        report.add(Finding(
            name=f"Bilateral {label} Asymmetry",
            severity=sev,
            description=f"left ({left:.1f}°) vs right ({right:.1f}°) {label.lower()} differ significantly",
            metric_value=round(ratio, 3),
            metric_label="asymmetry ratio",
        ))
