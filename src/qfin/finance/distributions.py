"""Probability distributions used by the representation layer."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

FloatArray = NDArray[np.float64]


class Distribution(Protocol):
    """Minimal distribution contract needed by ``qfin.encode``."""

    @property
    def mean(self) -> float: ...

    def cdf(self, x: ArrayLike) -> FloatArray: ...

    def ppf(self, q: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class Normal:
    """Normal distribution parameterized by mean and standard deviation."""

    mean_value: float = 0.0
    standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.mean_value):
            raise ValueError("mean_value must be finite")
        if not isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be finite and positive")

    @property
    def mean(self) -> float:
        return self.mean_value

    def cdf(self, x: ArrayLike) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        return np.asarray(ndtr((values - self.mean_value) / self.standard_deviation))

    def ppf(self, q: ArrayLike) -> FloatArray:
        probabilities = np.asarray(q, dtype=np.float64)
        return self.mean_value + self.standard_deviation * np.asarray(ndtri(probabilities))


@dataclass(frozen=True, slots=True)
class LogNormal:
    """Lognormal distribution where ``log(X) ~ Normal(mu, sigma)``."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")

    @property
    def mean(self) -> float:
        return exp(self.mu + 0.5 * self.sigma**2)

    def cdf(self, x: ArrayLike) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(values, dtype=np.float64)
        positive = values > 0
        result[positive] = ndtr((np.log(values[positive]) - self.mu) / self.sigma)
        return result

    def ppf(self, q: ArrayLike) -> FloatArray:
        probabilities = np.asarray(q, dtype=np.float64)
        return np.asarray(
            np.exp(self.mu + self.sigma * ndtri(probabilities)), dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class EmpiricalDistribution:
    """Finite empirical distribution with optional observation weights."""

    values: FloatArray
    probabilities: FloatArray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("values must contain at least one finite observation")
        if self.probabilities is None:
            probabilities = np.full(values.size, 1.0 / values.size)
        else:
            probabilities = np.asarray(self.probabilities, dtype=np.float64).reshape(-1)
            if probabilities.shape != values.shape:
                raise ValueError("probabilities must have the same shape as values")
            if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
                raise ValueError("probabilities must be finite and non-negative")
            total = float(np.sum(probabilities))
            if total <= 0:
                raise ValueError("probabilities must have positive total mass")
            probabilities = probabilities / total

        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_probabilities = probabilities[order]
        unique_values, inverse = np.unique(sorted_values, return_inverse=True)
        combined = np.zeros(unique_values.size, dtype=np.float64)
        np.add.at(combined, inverse, sorted_probabilities)
        unique_values.setflags(write=False)
        combined.setflags(write=False)
        object.__setattr__(self, "values", unique_values)
        object.__setattr__(self, "probabilities", combined)

    @property
    def mean(self) -> float:
        assert self.probabilities is not None
        return float(np.dot(self.values, self.probabilities))

    def cdf(self, x: ArrayLike) -> FloatArray:
        assert self.probabilities is not None
        query = np.asarray(x, dtype=np.float64)
        cumulative = np.cumsum(self.probabilities)
        indices = np.searchsorted(self.values, query, side="right") - 1
        result = np.zeros_like(query, dtype=np.float64)
        mask = indices >= 0
        result[mask] = cumulative[indices[mask]]
        return result

    def ppf(self, q: ArrayLike) -> FloatArray:
        assert self.probabilities is not None
        probabilities = np.asarray(q, dtype=np.float64)
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("quantiles must lie in [0, 1]")
        cumulative = np.cumsum(self.probabilities)
        indices = np.searchsorted(cumulative, probabilities, side="left")
        indices = np.clip(indices, 0, self.values.size - 1)
        return np.asarray(self.values[indices], dtype=np.float64)
