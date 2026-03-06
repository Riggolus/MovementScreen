"""Tests for compensation detection logic."""
import pytest

from movementscreen.analysis.joint_angles import JointAngles
from movementscreen.analysis.compensation import (
    CompensationReport,
    Finding,
    Severity,
    detect_compensations,
)


def _make_angles(**kwargs) -> JointAngles:
    angles = JointAngles()
    for k, v in kwargs.items():
        setattr(angles, k, v)
    return angles


# ---------------------------------------------------------------------------
# Trunk lean
# ---------------------------------------------------------------------------

def test_no_compensation_neutral_trunk():
    angles = _make_angles(trunk_lean_degrees=10.0)
    report = detect_compensations(angles)
    names = [f.name for f in report.findings]
    assert "Excessive Forward Trunk Lean" not in names


def test_mild_trunk_lean():
    angles = _make_angles(trunk_lean_degrees=25.0)
    report = detect_compensations(angles)
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Severity.MILD


def test_severe_trunk_lean():
    angles = _make_angles(trunk_lean_degrees=55.0)
    report = detect_compensations(angles)
    trunk_findings = [f for f in report.findings if "Trunk Lean" in f.name]
    assert trunk_findings
    assert trunk_findings[0].severity == Severity.SEVERE


# ---------------------------------------------------------------------------
# Lateral trunk shift
# ---------------------------------------------------------------------------

def test_no_lateral_shift():
    angles = _make_angles(lateral_trunk_shift=0.01)
    report = detect_compensations(angles)
    shift_findings = [f for f in report.findings if "Lateral" in f.name]
    assert not shift_findings


def test_moderate_lateral_shift():
    angles = _make_angles(lateral_trunk_shift=0.06)
    report = detect_compensations(angles)
    shift_findings = [f for f in report.findings if "Lateral" in f.name]
    assert shift_findings
    assert shift_findings[0].severity == Severity.MODERATE


# ---------------------------------------------------------------------------
# Bilateral asymmetry
# ---------------------------------------------------------------------------

def test_symmetric_knees_no_finding():
    angles = _make_angles(left_knee_flexion=90.0, right_knee_flexion=90.0)
    report = detect_compensations(angles)
    asym = [f for f in report.findings if "Asymmetry" in f.name and "Knee" in f.name]
    assert not asym


def test_asymmetric_knees_finding():
    angles = _make_angles(left_knee_flexion=90.0, right_knee_flexion=60.0)
    report = detect_compensations(angles)
    asym = [f for f in report.findings if "Knee Flexion Asymmetry" in f.name]
    assert asym
    assert asym[0].severity in (Severity.MILD, Severity.MODERATE, Severity.SEVERE)


# ---------------------------------------------------------------------------
# Worst severity
# ---------------------------------------------------------------------------

def test_worst_severity_empty():
    report = CompensationReport()
    assert report.worst_severity == Severity.NONE


def test_worst_severity_multiple():
    report = CompensationReport()
    report.add(Finding("A", Severity.MILD, "test"))
    report.add(Finding("B", Severity.SEVERE, "test"))
    report.add(Finding("C", Severity.MODERATE, "test"))
    assert report.worst_severity == Severity.SEVERE
