"""Public high-level spline API."""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import cuda as cuda_backend
from . import numpy as numpy_backend
from .compile import Extrapolation, SplineData, compile_spline


@dataclass(frozen=True)
class PackedSplineCollection:
    knots: np.ndarray
    coefficients: np.ndarray
    knot_offsets: np.ndarray
    coefficient_offsets: np.ndarray
    extrapolations: np.ndarray


def _pack_splines(data: Sequence[SplineData]) -> PackedSplineCollection:
    if not data:
        raise ValueError("at least one spline is required")

    knot_offsets = [0]
    coefficient_offsets = [0]
    knots = []
    coefficients = []
    extrapolations = []

    for spline in data:
        knots.append(spline.knots)
        coefficients.append(spline.coefficients)
        knot_offsets.append(knot_offsets[-1] + spline.knots.shape[0])
        coefficient_offsets.append(coefficient_offsets[-1] + spline.coefficients.shape[0])
        extrapolations.append(int(spline.extrapolation))

    return PackedSplineCollection(
        knots=np.concatenate(knots),
        coefficients=np.concatenate(coefficients, axis=0),
        knot_offsets=np.asarray(knot_offsets, dtype=np.int64),
        coefficient_offsets=np.asarray(coefficient_offsets, dtype=np.int64),
        extrapolations=np.asarray(extrapolations, dtype=np.int64),
    )


class Spline:
    def __init__(self, x, y, *, extrapolation: Extrapolation | str = Extrapolation.CLAMP) -> None:
        if isinstance(extrapolation, str):
            extrapolation = Extrapolation[extrapolation.upper()]
        self.data = compile_spline(x, y, extrapolation=Extrapolation(extrapolation))

    @property
    def knots(self):
        return self.data.knots

    @property
    def coefficients(self):
        return self.data.coefficients

    def to_device(self):
        return cuda_backend.to_device(self.data)

    def evaluate(self, x, *, threads_per_block: int = 256):
        return cuda_backend.evaluate(self.data, x, threads_per_block=threads_per_block)

    def to_numpy(self):
        return lambda x: numpy_backend.evaluate_spline(self.data, x)

    def __len__(self):
        return self.data.knots.shape[0]


class SplineCollection:
    def __init__(self, splines: Sequence[Spline]) -> None:
        self.splines = tuple(splines)
        self.data = _pack_splines([spline.data for spline in self.splines])

    def to_device(self):
        return cuda_backend.to_device_collection(self.data)

    def evaluate(self, spline_indices, x, *, threads_per_block: int = 256):
        return cuda_backend.evaluate_collection(
            self.data, spline_indices, x, threads_per_block=threads_per_block
        )

    def __len__(self):
        return len(self.splines)
