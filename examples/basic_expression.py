import numpy as np

from kernex import Expression


expression = Expression(
    "H0*k1/(k2 + k3) - H1*k3/(k1 - k2) - k0",
    parameter_order=["H0", "H1", "k0", "k1", "k2", "k3"],
)
parameters = np.random.uniform(0, 1, (100_000, 6))

gpu_result = expression.evaluate(parameters)
numpy_result = expression.to_numpy()(parameters)

print(np.allclose(gpu_result, numpy_result))
