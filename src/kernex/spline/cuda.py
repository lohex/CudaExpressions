"""CUDA batch execution for Kernex splines."""
import numpy as np
from numba import cuda

from .inline import evaluate_spline_collection_inline, evaluate_spline_inline


@cuda.jit
def _spline_kernel(knots, coefficients, extrapolation, query, result):
    index = cuda.grid(1)
    if index < query.shape[0]:
        result[index] = evaluate_spline_inline(
            knots, coefficients, query[index], extrapolation
        )


@cuda.jit
def _collection_kernel(
    knots,
    coefficients,
    knot_offsets,
    coefficient_offsets,
    extrapolations,
    spline_indices,
    query,
    result,
):
    index = cuda.grid(1)
    if index < query.shape[0]:
        result[index] = evaluate_spline_collection_inline(
            knots,
            coefficients,
            knot_offsets,
            coefficient_offsets,
            extrapolations,
            spline_indices[index],
            query[index],
        )


def to_device(data):
    return cuda.to_device(data.knots), cuda.to_device(data.coefficients), int(data.extrapolation)


def to_device_collection(packed):
    return (
        cuda.to_device(packed.knots),
        cuda.to_device(packed.coefficients),
        cuda.to_device(packed.knot_offsets),
        cuda.to_device(packed.coefficient_offsets),
        cuda.to_device(packed.extrapolations),
    )


def evaluate(data, x, *, threads_per_block: int = 256):
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")

    query = np.asarray(x, dtype=np.float64)
    scalar = query.ndim == 0
    flat = query.reshape(-1)
    if flat.size == 0:
        return np.empty(query.shape, dtype=np.float64)

    d_knots, d_coefficients, extrapolation = to_device(data)
    d_query = cuda.to_device(flat)
    d_result = cuda.device_array(flat.shape[0], dtype=np.float64)
    blocks = (flat.shape[0] + threads_per_block - 1) // threads_per_block
    _spline_kernel[blocks, threads_per_block](
        d_knots, d_coefficients, extrapolation, d_query, d_result
    )
    cuda.synchronize()
    result = d_result.copy_to_host().reshape(query.shape)
    if scalar:
        return float(result)
    return result


def evaluate_collection(packed, spline_indices, x, *, threads_per_block: int = 256):
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")

    indices = np.asarray(spline_indices, dtype=np.int64)
    query = np.asarray(x, dtype=np.float64)
    if indices.shape != query.shape:
        raise ValueError("spline_indices and x must have the same shape")
    if query.size == 0:
        return np.empty(query.shape, dtype=np.float64)

    shape = query.shape
    indices = indices.reshape(-1)
    query = query.reshape(-1)
    device_data = to_device_collection(packed)
    d_indices = cuda.to_device(indices)
    d_query = cuda.to_device(query)
    d_result = cuda.device_array(query.shape[0], dtype=np.float64)
    blocks = (query.shape[0] + threads_per_block - 1) // threads_per_block
    _collection_kernel[blocks, threads_per_block](
        *device_data,
        d_indices,
        d_query,
        d_result,
    )
    cuda.synchronize()
    return d_result.copy_to_host().reshape(shape)
