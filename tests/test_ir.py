import numpy as np

from kernex import Expression, OpCode


def test_ir_is_compact_int64_tensor():
    expression = Expression("a + b", parameter_order=["a", "b"])
    tensor = expression.tensor

    assert tensor.dtype == np.int64
    assert tensor.ndim == 2
    assert tensor.shape[1] == 3
    assert tensor[-1, 0] == int(OpCode.RETURN)
