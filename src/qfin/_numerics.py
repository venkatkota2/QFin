"""Focused numerical reductions for cancellation-sensitive financial aggregates."""

from __future__ import annotations

from math import fsum

import numpy as np
from numpy.typing import ArrayLike


def stable_sum(value: ArrayLike) -> float:
    """Use NumPy pairwise reduction, escalating only under severe cancellation."""

    values = np.asarray(value, dtype=np.float64).reshape(-1)
    total = float(np.sum(values, dtype=np.float64))
    if values.size < 2 or not (np.any(values < 0.0) and np.any(values > 0.0)):
        return total
    magnitude = float(np.sum(np.abs(values), dtype=np.float64))
    if magnitude and abs(total) <= 1.0e-10 * magnitude:
        return fsum(float(item) for item in values)
    return total


def stable_weighted_sum(values: ArrayLike, weights: ArrayLike) -> float:
    """Return a cancellation-aware sum of elementwise weighted values."""

    left = np.asarray(values, dtype=np.float64)
    right = np.asarray(weights, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("weighted-sum values and weights must have equal shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        products = left * right
    return stable_sum(products)


__all__ = ["stable_sum", "stable_weighted_sum"]
