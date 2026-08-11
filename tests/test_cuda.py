import numpy as np

from kernex import Expression, ExpressionVector


def test_cuda_scalar_matches_numpy():
    expression = Expression(
        "a*x + b",
        parameter_order=["a", "b"],
        variable_order=["x"],
    )
    parameters = np.asarray([[2.0, 1.0], [3.0, -1.0]])
    variables = np.asarray([[4.0], [2.0]])

    gpu_result = expression.evaluate(parameters, variables)
    numpy_result = expression.to_numpy()(parameters, variables)

    assert np.allclose(gpu_result, numpy_result)


def test_cuda_vector_matches_numpy():
    expression = ExpressionVector(
        ["a+b", "a*b", "a-b"],
        parameter_order=["a", "b"],
    )
    parameters = np.asarray([[2.0, 3.0], [4.0, 5.0]])

    gpu_result = expression.evaluate(parameters)
    numpy_result = expression.to_numpy()(parameters)

    assert np.allclose(gpu_result, numpy_result)
