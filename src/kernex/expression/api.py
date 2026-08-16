"""Public high-level expression API."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sympy as sp

from . import cuda as cuda_backend
from . import numpy as numpy_backend
from .parser import parse_expression, parse_expression_vector


class Expression:
    def __init__(
        self,
        expression: str | sp.Expr,
        parameter_order: Sequence[str | sp.Symbol] | None = None,
        variable_order: Sequence[str | sp.Symbol] | None = None,
        *,
        allow_unused: bool = True,
        optimize: bool = True,
    ) -> None:
        self.base_expression, self.parameters, self.variables, self.ir = parse_expression(
            expression,
            parameter_order,
            variable_order,
            allow_unused=allow_unused,
            optimize=optimize,
        )

    @property
    def tensor(self) -> np.ndarray:
        return self.ir.to_array()

    def to_device(self):
        return cuda_backend.to_device(self.tensor)

    def evaluate(
        self,
        parameters: np.ndarray,
        variables: np.ndarray | None = None,
        *,
        threads_per_block: int = 256,
    ) -> np.ndarray:
        return cuda_backend.evaluate_scalar(
            self.tensor, parameters, variables, threads_per_block=threads_per_block
        )

    def to_numpy(self, *, on_array: bool = True):
        return numpy_backend.scalar_function(
            self.base_expression, self.parameters, self.variables, on_array=on_array
        )

    def toTensor(self, on_device: bool = True):
        return self.to_device() if on_device else self.tensor

    def toNumpy(self, on_array: bool = True):
        return self.to_numpy(on_array=on_array)

    def eval(self, params, variables=None, threadsperblock: int = 256):
        return self.evaluate(params, variables, threads_per_block=threadsperblock)


class ExpressionVector:
    def __init__(
        self,
        expressions: Sequence[str | sp.Expr],
        parameter_order: Sequence[str | sp.Symbol] | None = None,
        variable_order: Sequence[str | sp.Symbol] | None = None,
        *,
        allow_unused: bool = True,
        optimize: bool = True,
    ) -> None:
        self.base_expressions, self.parameters, self.variables, self.ir = parse_expression_vector(
            expressions,
            parameter_order,
            variable_order,
            allow_unused=allow_unused,
            optimize=optimize,
        )

    @property
    def output_dim(self) -> int:
        return self.ir.output_dim

    @property
    def tensor(self) -> np.ndarray:
        return self.ir.to_array()

    def to_device(self):
        return cuda_backend.to_device(self.tensor)

    def evaluate(self, parameters, variables=None, *, threads_per_block: int = 256):
        return cuda_backend.evaluate_vector(
            self.tensor,
            self.output_dim,
            parameters,
            variables,
            threads_per_block=threads_per_block,
        )

    def to_numpy(self, *, on_array: bool = True):
        return numpy_backend.vector_function(
            self.base_expressions, self.parameters, self.variables, on_array=on_array
        )

    def toTensor(self, on_device: bool = True):
        return self.to_device() if on_device else self.tensor

    def toNumpy(self, on_array: bool = True):
        return self.to_numpy(on_array=on_array)

    def eval(self, params, variables=None, threadsperblock: int = 256):
        return self.evaluate(params, variables, threads_per_block=threadsperblock)
