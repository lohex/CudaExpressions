import numpy as np

from kernex import Expression, ExpressionVector


def test_scalar_numpy_backend():
    expression = Expression("a*x + b", parameter_order=["a", "b"], variable_order=["x"])
    parameters = np.asarray([[2.0, 1.0], [3.0, -1.0]])
    variables = np.asarray([[4.0], [2.0]])

    result = expression.to_numpy()(parameters, variables)

    assert np.allclose(result, [9.0, 5.0])


def test_vector_numpy_backend_has_batch_by_output_shape():
    expression = ExpressionVector(
        ["a + b", "a*b", "0"],
        parameter_order=["a", "b"],
    )
    parameters = np.asarray([[2.0, 3.0], [4.0, 5.0]])

    result = expression.to_numpy()(parameters)

    assert result.shape == (2, 3)
    assert np.allclose(result, [[5.0, 6.0, 0.0], [9.0, 20.0, 0.0]])
