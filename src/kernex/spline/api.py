"""Public high-level spline API."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import cuda as cuda_backend
from . import numpy as numpy_backend
from .collection import PackedSplineCollection, pack_splines
from .compile import Extrapolation, compile_spline


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
        self.data: PackedSplineCollection = pack_splines([spline.data for spline in self.splines])

    def evaluate(self, spline_indices, x, *, threads_per_block: int = 256):
        return cuda_backend.evaluate_collection(self.data, spline_indices, x, threads_per_block=threads_per_block)

    def __len__(self):
        return len(self.splines)
