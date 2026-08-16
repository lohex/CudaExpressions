"""Packed storage for heterogeneous spline collections."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .compile import SplineData


@dataclass(frozen=True)
class PackedSplineCollection:
    knots: np.ndarray
    coefficients: np.ndarray
    knot_offsets: np.ndarray
    coefficient_offsets: np.ndarray
    extrapolations: np.ndarray


def pack_splines(data: Sequence[SplineData]) -> PackedSplineCollection:
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
