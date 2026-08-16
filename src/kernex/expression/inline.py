"""Device-side expression evaluators for use inside user CUDA kernels."""
from numba import cuda

from .ir import OpCode, SymbolKind

_OP_CONSTANT = int(OpCode.CONSTANT)
_OP_SYMBOL = int(OpCode.SYMBOL)
_OP_ADD = int(OpCode.ADD)
_OP_MUL = int(OpCode.MUL)
_OP_POW = int(OpCode.POW)
_OP_STORE = int(OpCode.STORE)
_OP_RETURN = int(OpCode.RETURN)
_SYMBOL_PARAMETER = int(SymbolKind.PARAMETER)
_SYMBOL_VARIABLE = int(SymbolKind.VARIABLE)


@cuda.jit(device=True)
def evaluate_expression_inline(expression_ir, parameters, variables, buffer):
    """Evaluate one scalar expression from an existing CUDA kernel."""
    result = 0.0
    for index in range(expression_ir.shape[0]):
        opcode = expression_ir[index, 0]
        arg1 = expression_ir[index, 1]
        arg2 = expression_ir[index, 2]
        if opcode == _OP_CONSTANT:
            buffer[index] = arg1
        elif opcode == _OP_SYMBOL and arg2 == _SYMBOL_PARAMETER:
            buffer[index] = parameters[arg1]
        elif opcode == _OP_SYMBOL and arg2 == _SYMBOL_VARIABLE:
            buffer[index] = variables[arg1]
        elif opcode == _OP_ADD:
            buffer[index] = buffer[arg1] + buffer[arg2]
        elif opcode == _OP_MUL:
            buffer[index] = buffer[arg1] * buffer[arg2]
        elif opcode == _OP_POW:
            buffer[index] = buffer[arg1] ** buffer[arg2]
        elif opcode == _OP_RETURN:
            result = buffer[arg1]
            buffer[index] = result
    return result


@cuda.jit(device=True)
def evaluate_expression_vector_inline(expression_ir, parameters, variables, buffer, result_vector):
    """Evaluate a vector expression from an existing CUDA kernel."""
    for index in range(expression_ir.shape[0]):
        opcode = expression_ir[index, 0]
        arg1 = expression_ir[index, 1]
        arg2 = expression_ir[index, 2]
        if opcode == _OP_CONSTANT:
            buffer[index] = arg1
        elif opcode == _OP_SYMBOL and arg2 == _SYMBOL_PARAMETER:
            buffer[index] = parameters[arg1]
        elif opcode == _OP_SYMBOL and arg2 == _SYMBOL_VARIABLE:
            buffer[index] = variables[arg1]
        elif opcode == _OP_ADD:
            buffer[index] = buffer[arg1] + buffer[arg2]
        elif opcode == _OP_MUL:
            buffer[index] = buffer[arg1] * buffer[arg2]
        elif opcode == _OP_POW:
            buffer[index] = buffer[arg1] ** buffer[arg2]
        elif opcode == _OP_STORE:
            result_vector[arg2] = buffer[arg1]
