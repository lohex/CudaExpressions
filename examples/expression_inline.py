from numba import cuda
from kernex import Expression
from kernex.expression.inline import evaluate_expression_inline

expression = Expression("a*x+b", parameter_order=["a", "b"], variable_order=["x"])

@cuda.jit
def kernel(ir, parameters, variables, buffer, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = evaluate_expression_inline(ir, parameters[i], variables[i], buffer[i])
