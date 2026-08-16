import numpy as np
import pytest

from kernex import Expression, ExpressionVector
from kernex.expression.inline import evaluate_expression_inline


def test_expression_numpy_reference():
    expr = Expression("a*x + b", parameter_order=["a", "b"], variable_order=["x"])
    params = np.array([[2.0, 1.0], [3.0, -1.0]])
    variables = np.array([[4.0], [2.0]])
    np.testing.assert_allclose(expr.to_numpy()(params, variables), [9.0, 5.0])


def test_expression_vector_numpy_reference():
    expr = ExpressionVector(["a*x", "a+x"], parameter_order=["a"], variable_order=["x"])
    params = np.array([[2.0], [3.0]])
    variables = np.array([[4.0], [2.0]])
    np.testing.assert_allclose(expr.to_numpy()(params, variables), [[8.0, 6.0], [6.0, 5.0]])


def test_non_integer_literal_rejected():
    with pytest.raises(ValueError):
        Expression("0.5*x", variable_order=["x"])


def test_inline_name_is_exported():
    assert callable(evaluate_expression_inline)
