from kernex import ExpressionVector


def test_vector_common_subexpressions_are_deduplicated():
    unoptimized = ExpressionVector(
        ["a+b", "(a+b)*c"],
        parameter_order=["a", "b", "c"],
        optimize=False,
    )
    optimized = ExpressionVector(
        ["a+b", "(a+b)*c"],
        parameter_order=["a", "b", "c"],
        optimize=True,
    )

    assert len(optimized.ir.instructions) < len(unoptimized.ir.instructions)
