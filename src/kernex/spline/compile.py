"""Host-side compilation of natural cubic splines."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Extrapolation(IntEnum):
    CLAMP = 0
    SPLINE = 1
    NAN = 2


@dataclass(frozen=True)
class SplineData:
    knots: np.ndarray
    coefficients: np.ndarray
    extrapolation: Extrapolation


def _normalize_xy(x, y):
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must have the same length")
    if x_array.shape[0] < 2:
        raise ValueError("at least two spline knots are required")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x and y must contain only finite values")
    order = np.argsort(x_array, kind="stable")
    x_array = x_array[order]
    y_array = y_array[order]
    if np.any(np.diff(x_array) <= 0.0):
        raise ValueError("spline knots must be unique")
    return x_array, y_array


def _solve_tridiagonal(lower, diagonal, upper, rhs):
    n = diagonal.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    c_prime = np.empty(max(n - 1, 0), dtype=np.float64)
    d_prime = np.empty(n, dtype=np.float64)
    denom = diagonal[0]
    if denom == 0.0:
        raise np.linalg.LinAlgError("singular tridiagonal system")
    if n > 1:
        c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        if denom == 0.0:
            raise np.linalg.LinAlgError("singular tridiagonal system")
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom
    solution = np.empty(n, dtype=np.float64)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def compile_spline(x, y, *, extrapolation: Extrapolation = Extrapolation.CLAMP) -> SplineData:
    """Compile samples to local coefficients a + b*dx + c*dx**2 + d*dx**3."""
    knots, values = _normalize_xy(x, y)
    extrapolation = Extrapolation(extrapolation)
    h = np.diff(knots)
    slopes = np.diff(values) / h
    n = knots.shape[0]
    second = np.zeros(n, dtype=np.float64)
    if n > 2:
        lower = h[1:-1].copy()
        diagonal = 2.0 * (h[:-1] + h[1:])
        upper = h[1:-1].copy()
        rhs = 6.0 * (slopes[1:] - slopes[:-1])
        second[1:-1] = _solve_tridiagonal(lower, diagonal, upper, rhs)
    a = values[:-1]
    b = slopes - h * (2.0 * second[:-1] + second[1:]) / 6.0
    c = second[:-1] / 2.0
    d = (second[1:] - second[:-1]) / (6.0 * h)
    coefficients = np.column_stack((a, b, c, d))
    return SplineData(knots=knots, coefficients=coefficients, extrapolation=extrapolation)
