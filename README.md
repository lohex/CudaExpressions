# Kernex

Kernex provides compact mathematical runtime objects that are prepared on the host and evaluated inline inside Numba CUDA kernels.

The package currently contains two modules:

* `kernex.expression`: symbolic expressions compiled to a compact integer IR.
* `kernex.spline`: natural cubic splines compiled to knots and local polynomial coefficients.

## Inline expression evaluation

```python
from kernex.expression.inline import evaluate_expression_inline
```

## Inline spline evaluation

```python
from kernex.spline.inline import evaluate_spline_inline
```

Splines sort knots together with their values, reject duplicate knots, use explicit extrapolation behavior, preserve floating-point coefficients, and use packed storage for heterogeneous spline collections.
