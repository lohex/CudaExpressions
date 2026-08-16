"""CUDA batch execution for Kernex expressions."""
from __future__ import annotations

import numpy as np
from numba import cuda

from .inline import evaluate_expression_inline, evaluate_expression_vector_inline


@cuda.jit
def _scalar_kernel(parameters, variables, expression_ir, buffer, results):
    index = cuda.grid(1)
    if index < parameters.shape[0]:
        results[index] = evaluate_expression_inline(
            expression_ir, parameters[index], variables[index], buffer[index]
        )


@cuda.jit
def _vector_kernel(parameters, variables, expression_ir, buffer, results):
    index = cuda.grid(1)
    if index < parameters.shape[0]:
        evaluate_expression_vector_inline(
            expression_ir, parameters[index], variables[index], buffer[index], results[index]
        )


def to_device(ir_array: np.ndarray):
    return cuda.to_device(ir_array)


def evaluate_scalar(ir_array, parameters, variables=None, *, threads_per_block: int = 256):
    _validate_batch_inputs(parameters, variables, threads_per_block)
    if variables is None:
        variables = np.empty((parameters.shape[0], 0), dtype=parameters.dtype)
    d_parameters = cuda.to_device(parameters)
    d_variables = cuda.to_device(variables)
    d_ir = cuda.to_device(ir_array)
    d_buffer = cuda.device_array((parameters.shape[0], ir_array.shape[0]), dtype=np.float64)
    d_results = cuda.device_array(parameters.shape[0], dtype=np.float64)
    blocks = (parameters.shape[0] + threads_per_block - 1) // threads_per_block
    _scalar_kernel[blocks, threads_per_block](d_parameters, d_variables, d_ir, d_buffer, d_results)
    cuda.synchronize()
    return d_results.copy_to_host()


def evaluate_vector(ir_array, output_dim, parameters, variables=None, *, threads_per_block: int = 256):
    _validate_batch_inputs(parameters, variables, threads_per_block)
    if variables is None:
        variables = np.empty((parameters.shape[0], 0), dtype=parameters.dtype)
    d_parameters = cuda.to_device(parameters)
    d_variables = cuda.to_device(variables)
    d_ir = cuda.to_device(ir_array)
    d_buffer = cuda.device_array((parameters.shape[0], ir_array.shape[0]), dtype=np.float64)
    d_results = cuda.device_array((parameters.shape[0], output_dim), dtype=np.float64)
    d_results[:] = 0.0
    blocks = (parameters.shape[0] + threads_per_block - 1) // threads_per_block
    _vector_kernel[blocks, threads_per_block](d_parameters, d_variables, d_ir, d_buffer, d_results)
    cuda.synchronize()
    return d_results.copy_to_host()


def _validate_batch_inputs(parameters, variables, threads_per_block):
    if parameters.ndim != 2:
        raise ValueError("parameters must be a two-dimensional array")
    if variables is not None:
        if variables.ndim != 2:
            raise ValueError("variables must be a two-dimensional array")
        if variables.shape[0] != parameters.shape[0]:
            raise ValueError("parameters and variables must have the same number of rows")
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
