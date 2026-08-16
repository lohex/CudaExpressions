import numpy as np

from kernex import Spline, SplineCollection


def test_collection_uses_packed_storage_without_padding():
    s1 = Spline(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    s2 = Spline(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 0.0, 1.0]))
    collection = SplineCollection([s1, s2])
    assert collection.data.knots.shape[0] == 6
    assert collection.data.coefficients.shape[0] == 4
    np.testing.assert_array_equal(collection.data.knot_offsets, [0, 2, 6])
    np.testing.assert_array_equal(collection.data.coefficient_offsets, [0, 1, 4])
