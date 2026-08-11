"""NumPy reference backend for Kernex expressions."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp


def scalar_function(
    expression: sp.Expr,
    parameters: Sequence[sp.Symbol],
    variables: Sequence[sp.Symbol],
    *,
    on_array: bool = True,
) -> Callable:
    """Create a NumPy function for a scalar expression."""
    if variables:
        evaluate_one = sp.lambdify((parameters, variables), expression, "numpy")
    else:
        evaluate_one = sp.lambdify((parameters,), expression, "numpy")

    if not on_array:
        return evaluate_one

    if variables:
        def evaluate_array(parameter_values: np.ndarray, variable_values: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    evaluate_one(parameter_row, variable_row)
                    for parameter_row, variable_row in zip(parameter_values, variable_values)
                ]
            )
        return evaluate_array

    def evaluate_array(parameter_values: np.ndarray) -> np.ndarray:
        return np.asarray([evaluate_one(parameter_row) for parameter_row in parameter_values])

    return evaluate_array


def vector_function(
    expressions: Sequence[sp.Expr],
    parameters: Sequence[sp.Symbol],
    variables: Sequence[sp.Symbol],
    *,
    on_array: bool = True,
) -> Callable:
    """Create a NumPy function for a vector-valued expression."""
    matrix = sp.Matrix(expressions)

    if variables:
        evaluate_one = sp.lambdify((parameters, variables), matrix, "numpy")
    else:
        evaluate_one = sp.lambdify((parameters,), matrix, "numpy")

    if not on_array:
        return evaluate_one

    if variables:
        def evaluate_array(parameter_values: np.ndarray, variable_values: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    np.asarray(evaluate_one(parameter_row, variable_row), dtype=float).reshape(-1)
                    for parameter_row, variable_row in zip(parameter_values, variable_values)
                ]
            )
        return evaluate_array

    def evaluate_array(parameter_values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.asarray(evaluate_one(parameter_row), dtype=float).reshape(-1)
                for parameter_row in parameter_values
            ]
        )

    return evaluate_array
