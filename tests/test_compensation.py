"""Tests for compensation detection logic."""
import pytest

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.analysis.compensation import (
    CompensationReport,
    Finding,
    Grade,
    detect_compensations,
)


def _make_angles(**kwargs) -> JointAngles:
    angles = JointAngles()
    for k, v in kwargs.items():
        setattr(angles, k, v)
    return angles


# ---------------------------------------------------------------------------
# Trunk lean — sagittal camera
# ---------------------------------------------------------------------------

def test_no_compensation_neutral_trunk():
    angles = _make_angles(trunk_lean_degrees=10.0)
    report = detect_compensations(angles, camera_angle="lateral")
    names = [f.name for f in report.findings]
    assert "Excessive Forward Trunk Lean" not in names


def test_grade_b_trunk_lean():
    # 16° is above the B threshold (15°) but below C (20°)
    angles = _make_angles(trunk_lean_degrees=16.0)
    report = detect_compensations(angles, camera_angle="lateral")
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Grade.B


def test_grade_c_trunk_lean():
    angles = _make_angles(trunk_lean_degrees=25.0)
    report = detect_compensations(angles, camera_angle="lateral")
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Grade.C


def test_grade_d_trunk_lean():
    angles = _make_angles(trunk_lean_degrees=40.0)
    report = detect_compensations(angles, camera_angle="lateral")
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Grade.D


def test_grade_e_trunk_lean():
    angles = _make_angles(trunk_lean_degrees=47.0)
    report = detect_compensations(angles, camera_angle="lateral")
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Grade.E


def test_grade_f_trunk_lean():
    angles = _make_angles(trunk_lean_degrees=55.0)
    report = detect_compensations(angles, camera_angle="lateral")
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Grade.F


# ---------------------------------------------------------------------------
# Lateral trunk shift — frontal camera
# ---------------------------------------------------------------------------

def test_no_lateral_shift():
    angles = _make_angles(lateral_trunk_shift=0.01)
    report = detect_compensations(angles, camera_angle="anterior")
    shift_findings = [f for f in report.findings if "Lateral" in f.name]
    assert not shift_findings


def test_grade_c_lateral_shift():
    # 0.025 is above C threshold (0.02) but below D (0.05)
    angles = _make_angles(lateral_trunk_shift=0.025)
    report = detect_compensations(angles, camera_angle="anterior")
    shift_findings = [f for f in report.findings if "Lateral Trunk Shift" in f.name]
    assert shift_findings
    assert shift_findings[0].severity == Grade.C


def test_grade_d_lateral_shift():
    angles = _make_angles(lateral_trunk_shift=0.06)
    report = detect_compensations(angles, camera_angle="anterior")
    shift_findings = [f for f in report.findings if "Lateral Trunk Shift" in f.name]
    assert shift_findings
    assert shift_findings[0].severity == Grade.D


# ---------------------------------------------------------------------------
# Bilateral asymmetry — frontal camera
# ---------------------------------------------------------------------------

def test_symmetric_knees_no_finding():
    angles = _make_angles(left_knee_flexion=90.0, right_knee_flexion=90.0)
    report = detect_compensations(angles, camera_angle="anterior")
    asym = [f for f in report.findings if "Asymmetry" in f.name and "Knee" in f.name]
    assert not asym


def test_asymmetric_knees_finding():
    angles = _make_angles(left_knee_flexion=90.0, right_knee_flexion=60.0)
    report = detect_compensations(angles, camera_angle="anterior")
    asym = [f for f in report.findings if "Knee Flexion Asymmetry" in f.name]
    assert asym
    assert asym[0].severity in (Grade.B, Grade.C, Grade.D, Grade.E, Grade.F)


# ---------------------------------------------------------------------------
# Overall grade
# ---------------------------------------------------------------------------

def test_overall_grade_empty():
    report = CompensationReport()
    assert report.worst_severity == Grade.A


def test_overall_grade_worst_wins():
    report = CompensationReport()
    report.add(Finding("X", Grade.C, "test"))
    report.add(Finding("Y", Grade.F, "test"))
    report.add(Finding("Z", Grade.D, "test"))
    assert report.worst_severity == Grade.F


def test_grade_ordering():
    """Verify Grade.A < B < C < D < E < F in worst_severity logic."""
    from movementscreen.analysis.compensation import Grade
    order = [Grade.A, Grade.B, Grade.C, Grade.D, Grade.E, Grade.F]
    for i, g in enumerate(order):
        report = CompensationReport()
        report.add(Finding("x", g, "test"))
        assert report.worst_severity == g
