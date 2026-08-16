import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from kernex import Extrapolation, Spline
from kernex.spline.inline import evaluate_spline_inline


def test_matches_scipy_natural_cubic_spline():
    x = np.array([0.0, 0.5, 1.5, 2.0, 4.0])
    y = np.sin(x)
    spline = Spline(x, y, extrapolation=Extrapolation.SPLINE)
    query = np.linspace(-0.25, 4.25, 101)
    expected = CubicSpline(x, y, bc_type="natural", extrapolate=True)(query)
    np.testing.assert_allclose(spline.to_numpy()(query), expected, rtol=1e-11, atol=1e-11)


def test_unsorted_input_keeps_x_y_pairs():
    x = np.array([2.0, 0.0, 1.0])
    y = np.array([4.0, 0.0, 1.0])
    spline = Spline(x, y)
    np.testing.assert_array_equal(spline.knots, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(spline.to_numpy()(spline.knots), [0.0, 1.0, 4.0])


def test_duplicate_knots_rejected():
    with pytest.raises(ValueError, match="unique"):
        Spline(np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0]))


def test_extrapolation_modes():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    clamp = Spline(x, y, extrapolation="clamp")
    nan_spline = Spline(x, y, extrapolation="nan")
    np.testing.assert_allclose(clamp.to_numpy()(np.array([-1.0, 3.0])), [0.0, 0.0])
    assert np.isnan(nan_spline.to_numpy()(np.array([-1.0, 3.0]))).all()


def test_inline_name_is_exported():
    assert callable(evaluate_spline_inline)
