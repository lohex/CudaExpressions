"""Compile SymPy expressions into Kernex bytecode."""
from __future__ import annotations

from collections.abc import Sequence

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .ir import ExpressionIR, Instruction, OpCode, SymbolKind
from .optimize import optimize_ir


class ExpressionParseError(ValueError):
    """Raised when an expression cannot be represented by the Kernex IR."""


def _normalize_symbols(symbols: Sequence[str | sp.Symbol]) -> list[sp.Symbol]:
    return [sp.Symbol(str(symbol)) for symbol in symbols]


class _Compiler:
    def __init__(self, parameters: list[sp.Symbol], variables: list[sp.Symbol]) -> None:
        self.parameters = parameters
        self.variables = variables
        self.instructions: list[Instruction] = []

    def emit(self, instruction: Instruction) -> int:
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def compile(self, expression: sp.Expr) -> int:
        if isinstance(expression, sp.core.Number):
            integer_value = int(expression)
            if integer_value != expression:
                raise ExpressionParseError(
                    "Non-integer numeric literals are not supported by the compact int64 IR. "
                    "Provide them as parameters instead."
                )
            return self.emit(Instruction(OpCode.CONSTANT, integer_value, -1))

        if isinstance(expression, sp.core.Symbol):
            if expression in self.parameters:
                return self.emit(
                    Instruction(
                        OpCode.SYMBOL,
                        self.parameters.index(expression),
                        int(SymbolKind.PARAMETER),
                    )
                )
            if expression in self.variables:
                return self.emit(
                    Instruction(
                        OpCode.SYMBOL,
                        self.variables.index(expression),
                        int(SymbolKind.VARIABLE),
                    )
                )
            raise ExpressionParseError(f"Unknown symbol: {expression}")

        if isinstance(expression, sp.core.Add):
            return self._compile_nary(expression.args, OpCode.ADD)

        if isinstance(expression, sp.core.Mul):
            return self._compile_nary(expression.args, OpCode.MUL)

        if isinstance(expression, sp.core.Pow):
            left = self.compile(expression.args[0])
            right = self.compile(expression.args[1])
            return self.emit(Instruction(OpCode.POW, left, right))

        raise ExpressionParseError(
            f"Unsupported SymPy operation {expression.func.__name__!r} in expression {expression!s}."
        )

    def _compile_nary(self, arguments: tuple[sp.Expr, ...], opcode: OpCode) -> int:
        left = self.compile(arguments[0])
        for argument in arguments[1:]:
            right = self.compile(argument)
            left = self.emit(Instruction(opcode, left, right))
        return left


def _resolve_symbol_order(
    expressions: Sequence[sp.Expr],
    parameter_order: Sequence[str | sp.Symbol] | None,
    variable_order: Sequence[str | sp.Symbol] | None,
    allow_unused: bool,
) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    symbols: set[sp.Symbol] = set()
    for expression in expressions:
        symbols.update(expression.atoms(sp.Symbol))

    parameters = _normalize_symbols(parameter_order or [])
    variables = _normalize_symbols(variable_order or [])

    if not parameters and not variables:
        parameters = sorted(symbols, key=str)

    overlap = set(parameters).intersection(variables)
    if overlap:
        names = ", ".join(sorted(map(str, overlap)))
        raise ValueError(f"Symbols cannot be both parameters and variables: {names}")

    listed = set(parameters).union(variables)
    missing = symbols.difference(listed)
    if missing:
        names = ", ".join(sorted(map(str, missing)))
        raise ValueError(f"Symbols missing from parameter/variable ordering: {names}")

    if not allow_unused:
        unused = listed.difference(symbols)
        if unused:
            names = ", ".join(sorted(map(str, unused)))
            raise ValueError(f"Unused symbols in parameter/variable ordering: {names}")

    return parameters, variables


def parse_expression(
    expression: str | sp.Expr,
    parameter_order: Sequence[str | sp.Symbol] | None = None,
    variable_order: Sequence[str | sp.Symbol] | None = None,
    *,
    allow_unused: bool = True,
    optimize: bool = True,
) -> tuple[sp.Expr, list[sp.Symbol], list[sp.Symbol], ExpressionIR]:
    """Parse one scalar expression."""
    sympy_expression = parse_expr(expression) if isinstance(expression, str) else expression
    sympy_expression = sp.simplify(sympy_expression)

    parameters, variables = _resolve_symbol_order(
        [sympy_expression],
        parameter_order,
        variable_order,
        allow_unused,
    )
    compiler = _Compiler(parameters, variables)
    result_index = compiler.compile(sympy_expression)
    compiler.emit(Instruction(OpCode.RETURN, result_index, 0))

    ir = ExpressionIR(tuple(compiler.instructions), output_dim=1)
    if optimize:
        ir = optimize_ir(ir)

    return sympy_expression, parameters, variables, ir


def parse_expression_vector(
    expressions: Sequence[str | sp.Expr],
    parameter_order: Sequence[str | sp.Symbol] | None = None,
    variable_order: Sequence[str | sp.Symbol] | None = None,
    *,
    allow_unused: bool = True,
    optimize: bool = True,
) -> tuple[list[sp.Expr], list[sp.Symbol], list[sp.Symbol], ExpressionIR]:
    """Parse a vector-valued expression into one shared instruction stream."""
    if not expressions:
        raise ValueError("expressions must contain at least one component")

    sympy_expressions = [
        parse_expr(expression) if isinstance(expression, str) else expression
        for expression in expressions
    ]
    sympy_expressions = [sp.simplify(expression) for expression in sympy_expressions]

    parameters, variables = _resolve_symbol_order(
        sympy_expressions,
        parameter_order,
        variable_order,
        allow_unused,
    )
    compiler = _Compiler(parameters, variables)

    for output_index, expression in enumerate(sympy_expressions):
        if expression == 0:
            continue
        result_index = compiler.compile(expression)
        compiler.emit(Instruction(OpCode.STORE, result_index, output_index))

    ir = ExpressionIR(tuple(compiler.instructions), output_dim=len(sympy_expressions))
    if optimize:
        ir = optimize_ir(ir)

    return sympy_expressions, parameters, variables, ir
