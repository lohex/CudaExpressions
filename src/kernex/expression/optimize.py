"""Optimization passes for Kernex expression bytecode."""
from __future__ import annotations

from .ir import ExpressionIR, Instruction, OpCode


def eliminate_duplicate_instructions(ir: ExpressionIR) -> ExpressionIR:
    """Eliminate duplicate pure instructions and redirect later references.

    STORE and RETURN are retained because they encode observable outputs.
    """
    optimized: list[Instruction] = []
    old_to_new: dict[int, int] = {}
    seen: dict[Instruction, int] = {}

    for old_index, instruction in enumerate(ir.instructions):
        opcode = instruction.opcode

        if opcode in (OpCode.CONSTANT, OpCode.SYMBOL):
            normalized = instruction
        elif opcode in (OpCode.STORE, OpCode.RETURN):
            normalized = Instruction(opcode, old_to_new[instruction.arg1], instruction.arg2)
        else:
            normalized = Instruction(
                opcode,
                old_to_new[instruction.arg1],
                old_to_new[instruction.arg2],
            )

        if opcode not in (OpCode.STORE, OpCode.RETURN) and normalized in seen:
            old_to_new[old_index] = seen[normalized]
            continue

        new_index = len(optimized)
        optimized.append(normalized)
        old_to_new[old_index] = new_index

        if opcode not in (OpCode.STORE, OpCode.RETURN):
            seen[normalized] = new_index

    return ExpressionIR(tuple(optimized), output_dim=ir.output_dim)


def optimize_ir(ir: ExpressionIR) -> ExpressionIR:
    """Run the default optimization pipeline."""
    return eliminate_duplicate_instructions(ir)
