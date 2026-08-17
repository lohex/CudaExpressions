import numpy as np
from numba import cuda

from kernex import Expression
from kernex.expression.inline import evaluate_expression_inline

expression = Expression("a*x+b", parameter_order=["a", "b"], variable_order=["x"])


@cuda.jit
def kernel(bytecode, parameters, variables, workspace, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = evaluate_expression_inline(
            bytecode, parameters[i], variables[i], workspace[i]
        )


parameters = np.array([[2.0, 1.0], [3.0, -1.0]])
variables = np.array([[4.0], [2.0]])
workspace = cuda.device_array(
    (parameters.shape[0], expression.bytecode.shape[0]), dtype=np.float64
)
result = cuda.device_array(parameters.shape[0], dtype=np.float64)

kernel[1, 32](
    expression.to_device(),
    cuda.to_device(parameters),
    cuda.to_device(variables),
    workspace,
    result,
)
