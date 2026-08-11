# Kernex

**Runtime symbolic expressions for CUDA kernels.**

Kernex compiles symbolic expressions into compact integer bytecode that can be copied to GPU memory and interpreted from inside an already compiled Numba CUDA kernel. The expression therefore remains runtime data instead of becoming part of the kernel source.

## Install

```bash
pip install -e .
```

## Evaluate a symbolic expression

```python
import numpy as np
from kernex import Expression

expression = Expression("H0*k1/(k2 + k3) - H1*k3/(k1 - k2) - k0")
parameters = np.random.uniform(0, 1, (1_000_000, 6))

result = expression.evaluate(parameters)
```

If no explicit ordering is supplied, symbols are ordered alphabetically. For stable model interfaces, specify `parameter_order`.

## Evaluate inline inside your own kernel

This is Kernex's primary use case.

```python
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
def simulation(expression_ir, parameters, state, workspace, rates):
    i = cuda.grid(1)
    if i < parameters.shape[0]:
        rates[i] = eval_inline(
            expression_ir,
            parameters[i],
            state[i],
            workspace[i],
        )
```

The same compiled kernel can evaluate different expressions by receiving different Kernex bytecode.

## Vector expressions

```python
from kernex import ExpressionVector

rates = ExpressionVector(
    [
        "k1 / k4 + k3 / k2",
        "k4 / (k1*k2)",
        "k1 + k2 / (k1-k3)",
    ]
)
result = rates.evaluate(parameters)
```

Shared instructions are deduplicated in the IR.

## NumPy reference backend

```python
reference = expression.to_numpy()
expected = reference(parameters, variables)
```

This is useful for validation and testing against CUDA results.

## Architecture

```text
String / SymPy expression
          |
          v
        parser
          |
          v
   expression bytecode
          |
          v
      optimizer
       /     \
      v       v
 NumPy ref   CUDA runtime interpreter
                 |
                 v
          eval_inline(...)
          inside user kernel
```

## Current expression support

The compact IR supports symbols, integer constants, addition, multiplication and powers. Division and subtraction are represented by SymPy through multiplication and powers. Non-integer numeric literals should currently be supplied as parameters.
