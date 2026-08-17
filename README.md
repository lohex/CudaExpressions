# Kernex

Kernex provides compact mathematical runtime objects that are prepared on the host and evaluated directly inside Numba CUDA kernels.

The central idea is to separate expensive or symbolic preprocessing from repeated device-side evaluation:

```text
host input
    ↓
compile / preprocess once
    ↓
compact device representation
    ↓
evaluate inline inside arbitrary CUDA kernels
```

This is useful when a mathematical object is part of a larger GPU simulation or algorithm and should be evaluated without launching a separate kernel or recompiling device code for every model instance.

Kernex currently provides two modules:

* `kernex.expression`: symbolic expressions compiled to compact integer bytecode.
* `kernex.spline`: natural cubic splines compiled to knots and local cubic polynomial coefficients.

## Installation

For development installs:

```bash
pip install -e .
```

Kernex currently depends on NumPy, Numba and SymPy. SciPy is only used for numerical validation in the test suite.

## Expressions

### Basic evaluation

```python
from kernex import Expression

expr = Expression(
    "k1 * x / (k2 + x)",
    parameter_order=["k1", "k2"],
    variable_order=["x"],
)

# Array-like inputs are normalized to float64 internally.
parameters = [
    [2.0, 1.0],
    [3.0, 0.5],
]
variables = [
    [4.0],
    [2.0],
]

result = expr.evaluate(parameters, variables)
```

Expressions are parsed with SymPy and compiled to a compact instruction stream available as `expr.bytecode`. The bytecode can be transferred to the device once and interpreted repeatedly from CUDA device code.

### Inline expression evaluation

The main use case is embedding an expression inside an existing kernel:

```python
import numpy as np
from numba import cuda

from kernex import Expression
from kernex.expression.inline import evaluate_expression_inline

expr = Expression(
    "k1 * x / (k2 + x)",
    parameter_order=["k1", "k2"],
    variable_order=["x"],
)

device_bytecode = expr.to_device()

parameters = np.array([[2.0, 1.0], [3.0, 0.5]])
variables = np.array([[4.0], [2.0]])

d_parameters = cuda.to_device(parameters)
d_variables = cuda.to_device(variables)
d_workspace = cuda.device_array(
    (len(parameters), expr.bytecode.shape[0]), dtype=np.float64
)
d_result = cuda.device_array(len(parameters), dtype=np.float64)


@cuda.jit
def simulation_kernel(bytecode, parameters, variables, workspace, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        rate = evaluate_expression_inline(
            bytecode,
            parameters[i],
            variables[i],
            workspace[i],
        )

        # Continue a larger device-side calculation without another kernel launch.
        result[i] = 2.0 * rate


threads = 128
blocks = (len(parameters) + threads - 1) // threads
simulation_kernel[blocks, threads](
    device_bytecode,
    d_parameters,
    d_variables,
    d_workspace,
    d_result,
)
```

The expression itself is runtime data. Different expression bytecode can therefore be passed to the same compiled CUDA kernel. `workspace` is scratch memory for intermediate instruction results, not an output buffer.

For vector-valued expressions, Kernex also provides:

```python
from kernex.expression.inline import evaluate_expression_vector_inline
```

## Splines

### Basic evaluation

```python
import numpy as np

from kernex import Spline

x = np.array([0.0, 1.0, 2.0, 4.0])
y = np.array([0.0, 0.8, 0.2, 1.0])

spline = Spline(x, y, extrapolation="clamp")
query = np.linspace(0.0, 4.0, 100)
result = spline.evaluate(query)
```

Spline compilation is performed on the host. Kernex sorts knots together with their values, rejects duplicate or non-finite knots, solves the natural cubic spline system and stores each interval as local cubic coefficients in Horner-friendly form.

Supported extrapolation policies are:

* `CLAMP`: evaluate at the nearest boundary value outside the knot range.
* `SPLINE`: continue the first or last cubic interval.
* `NAN`: return `NaN` outside the knot range.

### Inline spline evaluation

A spline can also be evaluated directly from a custom CUDA kernel:

```python
import numpy as np
from numba import cuda

from kernex import Spline
from kernex.spline.inline import evaluate_spline_inline

spline = Spline(
    np.array([0.0, 1.0, 2.0, 4.0]),
    np.array([0.0, 0.8, 0.2, 1.0]),
    extrapolation="clamp",
)

d_knots, d_coefficients, extrapolation = spline.to_device()
query = np.linspace(0.0, 4.0, 1024)
d_query = cuda.to_device(query)
d_result = cuda.device_array(query.shape[0], dtype=np.float64)


@cuda.jit
def simulation_kernel(knots, coefficients, extrapolation, query, result):
    i = cuda.grid(1)
    if i < query.shape[0]:
        forcing = evaluate_spline_inline(
            knots,
            coefficients,
            query[i],
            extrapolation,
        )

        result[i] = forcing * forcing


threads = 128
blocks = (query.shape[0] + threads - 1) // threads
simulation_kernel[blocks, threads](
    d_knots,
    d_coefficients,
    extrapolation,
    d_query,
    d_result,
)
```

For heterogeneous sets of splines, `SplineCollection` uses packed storage with offsets instead of padding. Inline collection evaluation is available through:

```python
from kernex.spline.inline import evaluate_spline_collection_inline
```

## Package layout

Kernex is organized by mathematical runtime object. Each module owns its host representation, NumPy/reference implementation, batch CUDA execution and inline device functions:

```text
kernex/
├── expression/
│   ├── api.py
│   ├── parser.py
│   ├── ir.py
│   ├── optimize.py
│   ├── numpy.py
│   ├── cuda.py
│   └── inline.py
│
└── spline/
    ├── api.py
    ├── compile.py
    ├── numpy.py
    ├── cuda.py
    └── inline.py
```

The explicit inline entry points are:

```python
from kernex.expression.inline import evaluate_expression_inline
from kernex.spline.inline import evaluate_spline_inline
```

This naming convention is intended to remain unambiguous as further runtime objects are added.

## Input and empty-batch behavior

Expression batch inputs are accepted as array-like objects and normalized to `float64`. Empty batches return correctly shaped empty NumPy arrays without attempting a zero-block CUDA launch. Spline batch evaluation follows the same empty-input convention.

## Testing

Expression tests compare parser, IR, NumPy and CUDA behavior. Spline tests validate the natural cubic spline implementation against `scipy.interpolate.CubicSpline(..., bc_type="natural")` and cover input ordering, duplicate knots, extrapolation and packed collections.

CUDA device functions can also be tested without physical CUDA hardware using Numba's CUDA simulator:

```bash
NUMBA_ENABLE_CUDASIM=1 pytest
```

## Project direction

Kernex is intended as a small family of host-prepared mathematical objects that can be passed to generic CUDA kernels as runtime data. Expressions and splines are the first two implementations of that model. Future modules can follow the same pattern when they benefit from compact device-side representation and low-overhead inline evaluation.
