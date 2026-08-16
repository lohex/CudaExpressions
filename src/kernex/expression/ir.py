"""Intermediate representation used by Kernex."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class OpCode(IntEnum):
    """Opcodes understood by the CUDA expression interpreter."""

    CONSTANT = 1
    SYMBOL = 2
    ADD = 3
    MUL = 4
    POW = 5
    STORE = 6
    RETURN = 7


class SymbolKind(IntEnum):
    """Source of a symbol value."""

    PARAMETER = 1
    VARIABLE = 2


@dataclass(frozen=True)
class Instruction:
    """One three-word instruction in the Kernex IR."""

    opcode: OpCode
    arg1: int
    arg2: int

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.opcode), self.arg1, self.arg2


@dataclass(frozen=True)
class ExpressionIR:
    """Linear expression bytecode suitable for transfer to a CUDA device."""

    instructions: tuple[Instruction, ...]
    output_dim: int = 1

    def to_array(self) -> np.ndarray:
        return np.asarray([instruction.as_tuple() for instruction in self.instructions], dtype=np.int64)
