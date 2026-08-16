from numba import cuda
from kernex import Spline
from kernex.spline.inline import evaluate_spline_inline

# Define x and y before creating the spline.
spline = Spline(x, y)
knots, coefficients, extrapolation = spline.to_device()

@cuda.jit
def kernel(knots, coefficients, extrapolation, query, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = evaluate_spline_inline(knots, coefficients, query[i], extrapolation)
