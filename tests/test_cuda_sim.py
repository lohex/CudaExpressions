import numpy as np
from numba import cuda

from kernex import Expression, Spline, SplineCollection
from kernex.expression.inline import evaluate_expression_inline
from kernex.spline.inline import evaluate_spline_inline, evaluate_spline_collection_inline


@cuda.jit
def expression_kernel(ir, params, variables, buffer, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = evaluate_expression_inline(ir, params[i], variables[i], buffer[i])


@cuda.jit
def spline_kernel(knots, coefficients, extrapolation, query, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = evaluate_spline_inline(knots, coefficients, query[i], extrapolation)


@cuda.jit
def collection_kernel(knots, coefficients, ko, co, ext, indices, query, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = evaluate_spline_collection_inline(knots, coefficients, ko, co, ext, indices[i], query[i])


def test_expression_inline_cuda_simulator():
    expr = Expression("a*x+b", parameter_order=["a", "b"], variable_order=["x"])
    params = np.array([[2.0, 1.0], [3.0, -1.0]])
    variables = np.array([[4.0], [2.0]])
    ir = cuda.to_device(expr.tensor)
    buffer = cuda.device_array((2, expr.tensor.shape[0]), dtype=np.float64)
    result = cuda.device_array(2, dtype=np.float64)
    expression_kernel[1, 32](ir, cuda.to_device(params), cuda.to_device(variables), buffer, result)
    np.testing.assert_allclose(result.copy_to_host(), [9.0, 5.0])


def test_spline_inline_cuda_simulator():
    spline = Spline(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]))
    knots, coefficients, extrapolation = spline.to_device()
    query = np.array([0.0, 0.5, 1.5, 3.0])
    result = cuda.device_array(query.shape[0], dtype=np.float64)
    spline_kernel[1, 32](knots, coefficients, extrapolation, cuda.to_device(query), result)
    np.testing.assert_allclose(result.copy_to_host(), spline.to_numpy()(query))


def test_spline_collection_inline_cuda_simulator():
    s1 = Spline(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    s2 = Spline(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 1.0]))
    packed = SplineCollection([s1, s2]).data
    query = np.array([0.25, 1.5])
    indices = np.array([0, 1], dtype=np.int64)
    result = cuda.device_array(2, dtype=np.float64)
    collection_kernel[1, 32](
        cuda.to_device(packed.knots), cuda.to_device(packed.coefficients),
        cuda.to_device(packed.knot_offsets), cuda.to_device(packed.coefficient_offsets),
        cuda.to_device(packed.extrapolations), cuda.to_device(indices), cuda.to_device(query), result,
    )
    expected = np.array([s1.to_numpy()(query[0]), s2.to_numpy()(query[1])])
    np.testing.assert_allclose(result.copy_to_host(), expected)
