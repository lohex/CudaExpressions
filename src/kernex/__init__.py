"""Kernex: runtime symbolic expressions for CUDA kernels."""

from .cuda import eval_inline, eval_vector_inline
from .expression import Expression, ExpressionVector, GPUExpression, GPUExpressionVector
from .ir import ExpressionIR, Instruction, OpCode, SymbolKind

__all__ = [
    "Expression",
    "ExpressionVector",
    "GPUExpression",
    "GPUExpressionVector",
    "ExpressionIR",
    "Instruction",
    "OpCode",
    "SymbolKind",
    "eval_inline",
    "eval_vector_inline",
]

__version__ = "0.1.0"
