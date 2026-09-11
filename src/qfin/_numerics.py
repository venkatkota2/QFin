"""Focused numerical reductions for cancellation-sensitive financial aggregates."""

from __future__ import annotations

from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.exceptions import QFinValidationError


def normalize_weights(value: ArrayLike) -> NDArray[np.float64]:
    """Normalize validated nonnegative weights without unnecessary scaling loss.

    Scale only when summing the original weights would overflow. Always scaling
    can move an exact CDF atom (e.g. weights 9:1 at 90%) across the VaR boundary.
    """

    weights = np.asarray(value, dtype=np.float64)
    with np.errstate(over="ignore"):
        total = float(np.sum(weights))
    if not np.isfinite(total):
        weights = weights / float(np.max(weights))
        total = float(np.sum(weights))
    return np.asarray(weights / total, dtype=np.float64)


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
        raise QFinValidationError("weighted-sum values and weights must have equal shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        products = left * right
    return stable_sum(products)


__all__ = ["stable_sum", "stable_weighted_sum"]
