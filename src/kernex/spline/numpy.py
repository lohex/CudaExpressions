"""NumPy reference backend for Kernex splines."""
from __future__ import annotations

import numpy as np

from .compile import Extrapolation, SplineData


def evaluate_spline(data: SplineData, x):
    query = np.asarray(x, dtype=np.float64)
    scalar = query.ndim == 0
    flat = query.reshape(-1).copy()
    outside = (flat < data.knots[0]) | (flat > data.knots[-1])
    if data.extrapolation == Extrapolation.CLAMP:
        flat = np.clip(flat, data.knots[0], data.knots[-1])
    indices = np.searchsorted(data.knots, flat, side="right") - 1
    indices = np.clip(indices, 0, data.coefficients.shape[0] - 1)
    dx = flat - data.knots[indices]
    coef = data.coefficients[indices]
    result = coef[:, 0] + dx * (coef[:, 1] + dx * (coef[:, 2] + dx * coef[:, 3]))
    if data.extrapolation == Extrapolation.NAN:
        result[outside] = np.nan
    result = result.reshape(query.shape)
    return float(result) if scalar else result
