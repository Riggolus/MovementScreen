"""Tests for geometry utilities."""
import numpy as np
import pytest

from movementscreen.utils.geometry import (
    angle_between,
    asymmetry_ratio,
    horizontal_offset,
    midpoint,
    vertical_angle,
)


def test_angle_between_90_degrees():
    a = np.array([1.0, 0.0])
    vertex = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(angle_between(a, vertex, b) - 90.0) < 1e-6


def test_angle_between_180_degrees():
    a = np.array([-1.0, 0.0])
    vertex = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert abs(angle_between(a, vertex, b) - 180.0) < 1e-6


def test_angle_between_0_degrees():
    a = np.array([1.0, 0.0])
    vertex = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    assert abs(angle_between(a, vertex, b) - 0.0) < 1e-6


def test_angle_between_3d():
    a = np.array([1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(angle_between(a, vertex, b) - 90.0) < 1e-6


def test_midpoint_two_points():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 4.0])
    result = midpoint(a, b)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0])


def test_midpoint_three_points():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 0.0])
    c = np.array([0.0, 3.0])
    result = midpoint(a, b, c)
    np.testing.assert_array_almost_equal(result, [1.0, 1.0])


def test_asymmetry_ratio_symmetric():
    assert asymmetry_ratio(10.0, 10.0) == 0.0


def test_asymmetry_ratio_fully_asymmetric():
    ratio = asymmetry_ratio(10.0, 0.0)
    assert abs(ratio - 1.0) < 1e-6


def test_asymmetry_ratio_partial():
    ratio = asymmetry_ratio(100.0, 80.0)
    expected = 20.0 / 100.0
    assert abs(ratio - expected) < 1e-6


def test_horizontal_offset():
    a = np.array([0.6, 0.5])
    b = np.array([0.4, 0.5])
    assert abs(horizontal_offset(a, b) - 0.2) < 1e-6
