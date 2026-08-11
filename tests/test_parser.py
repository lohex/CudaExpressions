import pytest

from kernex import Expression, OpCode


def test_parameter_order_defaults_to_sorted_symbols():
    expression = Expression("b + a")
    assert [str(symbol) for symbol in expression.parameters] == ["a", "b"]


def test_parameters_and_variables_are_separate():
    expression = Expression(
        "a*x + b",
        parameter_order=["a", "b"],
        variable_order=["x"],
    )
    assert [str(symbol) for symbol in expression.parameters] == ["a", "b"]
    assert [str(symbol) for symbol in expression.variables] == ["x"]


def test_non_integer_literal_is_rejected():
    with pytest.raises(ValueError, match="Non-integer"):
        Expression("0.5*x", parameter_order=["x"])


def test_scalar_ir_ends_with_return():
    expression = Expression("a + b", parameter_order=["a", "b"])
    assert expression.ir.instructions[-1].opcode == OpCode.RETURN
