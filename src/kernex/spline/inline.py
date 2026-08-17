"""Device-side spline evaluators for use inside user CUDA kernels."""
import math

from numba import cuda

from .compile import Extrapolation

_EXTRAP_CLAMP = int(Extrapolation.CLAMP)
_EXTRAP_NAN = int(Extrapolation.NAN)


@cuda.jit(device=True)
def _find_interval(knots, x, start, end):
    low = start
    high = end - 1
    while low + 1 < high:
        mid = (low + high) // 2
        if x < knots[mid]:
            high = mid
        else:
            low = mid
    return low


@cuda.jit(device=True)
def _evaluate_interval(knots, coefficients, knot_index, coefficient_index, x):
    dx = x - knots[knot_index]
    a = coefficients[coefficient_index, 0]
    b = coefficients[coefficient_index, 1]
    c = coefficients[coefficient_index, 2]
    d = coefficients[coefficient_index, 3]
    return a + dx * (b + dx * (c + dx * d))


@cuda.jit(device=True)
def evaluate_spline_inline(knots, coefficients, x, extrapolation):
    """Evaluate one spline from inside an existing CUDA kernel."""
    first = knots[0]
    last = knots[knots.shape[0] - 1]
    if x < first:
        if extrapolation == _EXTRAP_NAN:
            return math.nan
        if extrapolation == _EXTRAP_CLAMP:
            x = first
    elif x > last:
        if extrapolation == _EXTRAP_NAN:
            return math.nan
        if extrapolation == _EXTRAP_CLAMP:
            x = last
    interval = _find_interval(knots, x, 0, knots.shape[0])
    if interval >= coefficients.shape[0]:
        interval = coefficients.shape[0] - 1
    return _evaluate_interval(knots, coefficients, interval, interval, x)


@cuda.jit(device=True)
def evaluate_spline_collection_inline(knots, coefficients, knot_offsets, coefficient_offsets, extrapolations, spline_index, x):
    knot_start = knot_offsets[spline_index]
    knot_end = knot_offsets[spline_index + 1]
    coefficient_start = coefficient_offsets[spline_index]
    coefficient_end = coefficient_offsets[spline_index + 1]
    first = knots[knot_start]
    last = knots[knot_end - 1]
    extrapolation = extrapolations[spline_index]
    if x < first:
        if extrapolation == _EXTRAP_NAN:
            return math.nan
        if extrapolation == _EXTRAP_CLAMP:
            x = first
    elif x > last:
        if extrapolation == _EXTRAP_NAN:
            return math.nan
        if extrapolation == _EXTRAP_CLAMP:
            x = last
    interval = _find_interval(knots, x, knot_start, knot_end)
    coefficient_index = coefficient_start + interval - knot_start
    if coefficient_index >= coefficient_end:
        coefficient_index = coefficient_end - 1
        interval = knot_end - 2
    return _evaluate_interval(knots, coefficients, interval, coefficient_index, x)
