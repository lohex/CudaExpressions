import numpy as np

from kernex import ExpressionVector


expression = ExpressionVector(
    ["k4/(k1*k2)", "k1/k4+k3/k2", "k1+k2/(k1-k3)", "k4+k2"],
    parameter_order=["k1", "k2", "k3", "k4"],
)
parameters = np.random.uniform(0, 1, (100_000, 4))

gpu_result = expression.evaluate(parameters)
numpy_result = expression.to_numpy()(parameters)

print(np.allclose(gpu_result, numpy_result))
