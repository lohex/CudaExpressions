import numpy as np
from numba import cuda

from kernex import Expression, eval_inline


expression = Expression(
    "k1*x/(k2 + x)",
    parameter_order=["k1", "k2"],
    variable_order=["x"],
)
device_ir = expression.to_device()


@cuda.jit
def kernel(expression_ir, parameters, variables, workspace, results):
    i = cuda.grid(1)
    if i < parameters.shape[0]:
        results[i] = eval_inline(
            expression_ir,
            parameters[i],
            variables[i],
            workspace[i],
        )


n = 100_000
parameters = np.random.uniform(0, 1, (n, 2))
variables = np.random.uniform(0, 1, (n, 1))
workspace = np.empty((n, expression.tensor.shape[0]))
results = np.empty(n)

kernel[(n + 255) // 256, 256](
    device_ir,
    cuda.to_device(parameters),
    cuda.to_device(variables),
    cuda.to_device(workspace),
    cuda.to_device(results),
)
