"""CUDA execution backend and device-side Kernex interpreter."""
from __future__ import annotations

import numpy as np
from numba import cuda

from .ir import OpCode, SymbolKind


@cuda.jit(device=True)
def eval_inline(expression_ir, parameters, variables, buffer):
    """Evaluate scalar Kernex bytecode from inside an existing CUDA kernel."""
    result = 0.0
    for index in range(expression_ir.shape[0]):
        opcode = expression_ir[index, 0]
        arg1 = expression_ir[index, 1]
        arg2 = expression_ir[index, 2]

        if opcode == int(OpCode.CONSTANT):
            buffer[index] = arg1
        elif opcode == int(OpCode.SYMBOL) and arg2 == int(SymbolKind.PARAMETER):
            buffer[index] = parameters[arg1]
        elif opcode == int(OpCode.SYMBOL) and arg2 == int(SymbolKind.VARIABLE):
            buffer[index] = variables[arg1]
        elif opcode == int(OpCode.ADD):
            buffer[index] = buffer[arg1] + buffer[arg2]
        elif opcode == int(OpCode.MUL):
            buffer[index] = buffer[arg1] * buffer[arg2]
        elif opcode == int(OpCode.POW):
            buffer[index] = buffer[arg1] ** buffer[arg2]
        elif opcode == int(OpCode.RETURN):
            result = buffer[arg1]
            buffer[index] = result

    return result


@cuda.jit(device=True)
def eval_vector_inline(expression_ir, parameters, variables, buffer, result_vector):
    """Evaluate vector-valued Kernex bytecode inside an existing CUDA kernel."""
    for index in range(expression_ir.shape[0]):
        opcode = expression_ir[index, 0]
        arg1 = expression_ir[index, 1]
        arg2 = expression_ir[index, 2]

        if opcode == int(OpCode.CONSTANT):
            buffer[index] = arg1
        elif opcode == int(OpCode.SYMBOL) and arg2 == int(SymbolKind.PARAMETER):
            buffer[index] = parameters[arg1]
        elif opcode == int(OpCode.SYMBOL) and arg2 == int(SymbolKind.VARIABLE):
            buffer[index] = variables[arg1]
        elif opcode == int(OpCode.ADD):
            buffer[index] = buffer[arg1] + buffer[arg2]
        elif opcode == int(OpCode.MUL):
            buffer[index] = buffer[arg1] * buffer[arg2]
        elif opcode == int(OpCode.POW):
            buffer[index] = buffer[arg1] ** buffer[arg2]
        elif opcode == int(OpCode.STORE):
            result_vector[arg2] = buffer[arg1]


@cuda.jit
def _scalar_kernel(parameters, variables, expression_ir, buffer, results):
    index = cuda.grid(1)
    if index < parameters.shape[0]:
        results[index] = eval_inline(
            expression_ir,
            parameters[index],
            variables[index],
            buffer[index],
        )


@cuda.jit
def _vector_kernel(parameters, variables, expression_ir, buffer, results):
    index = cuda.grid(1)
    if index < parameters.shape[0]:
        eval_vector_inline(
            expression_ir,
            parameters[index],
            variables[index],
            buffer[index],
            results[index],
        )


def to_device(ir_array: np.ndarray):
    """Copy an IR array to the active CUDA device."""
    return cuda.to_device(ir_array)


def evaluate_scalar(
    ir_array: np.ndarray,
    parameters: np.ndarray,
    variables: np.ndarray | None = None,
    *,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Evaluate a scalar expression for a batch of parameter rows."""
    _validate_batch_inputs(parameters, variables, threads_per_block)

    if variables is None:
        variables = np.empty((parameters.shape[0], 0), dtype=parameters.dtype)

    device_parameters = cuda.to_device(parameters)
    device_variables = cuda.to_device(variables)
    device_ir = cuda.to_device(ir_array)
    device_buffer = cuda.device_array((parameters.shape[0], ir_array.shape[0]), dtype=np.float64)
    device_results = cuda.device_array(parameters.shape[0], dtype=np.float64)

    blocks_per_grid = (parameters.shape[0] + threads_per_block - 1) // threads_per_block
    _scalar_kernel[blocks_per_grid, threads_per_block](
        device_parameters,
        device_variables,
        device_ir,
        device_buffer,
        device_results,
    )
    cuda.synchronize()
    return device_results.copy_to_host()


def evaluate_vector(
    ir_array: np.ndarray,
    output_dim: int,
    parameters: np.ndarray,
    variables: np.ndarray | None = None,
    *,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Evaluate a vector expression for a batch of parameter rows."""
    _validate_batch_inputs(parameters, variables, threads_per_block)

    if variables is None:
        variables = np.empty((parameters.shape[0], 0), dtype=parameters.dtype)

    device_parameters = cuda.to_device(parameters)
    device_variables = cuda.to_device(variables)
    device_ir = cuda.to_device(ir_array)
    device_buffer = cuda.device_array((parameters.shape[0], ir_array.shape[0]), dtype=np.float64)
    device_results = cuda.device_array((parameters.shape[0], output_dim), dtype=np.float64)
    device_results[:] = 0.0

    blocks_per_grid = (parameters.shape[0] + threads_per_block - 1) // threads_per_block
    _vector_kernel[blocks_per_grid, threads_per_block](
        device_parameters,
        device_variables,
        device_ir,
        device_buffer,
        device_results,
    )
    cuda.synchronize()
    return device_results.copy_to_host()


def _validate_batch_inputs(
    parameters: np.ndarray,
    variables: np.ndarray | None,
    threads_per_block: int,
) -> None:
    if parameters.ndim != 2:
        raise ValueError("parameters must be a two-dimensional array")
    if variables is not None:
        if variables.ndim != 2:
            raise ValueError("variables must be a two-dimensional array")
        if variables.shape[0] != parameters.shape[0]:
            raise ValueError("parameters and variables must have the same number of rows")
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
