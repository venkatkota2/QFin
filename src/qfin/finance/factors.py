"""Explicit dependence assumptions for multi-factor loss scenarios."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.finance.risk import LossDistribution

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FactorScenarios:
    """A scenario matrix with named factors and an explicit dependence label."""

    values: FloatArray
    factor_names: Sequence[str]
    dependence_assumption: str

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float64)
        names = tuple(self.factor_names)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("values must be a non-empty scenario-by-factor matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("factor scenarios must be finite")
        if len(names) != values.shape[1]:
            raise ValueError("factor_names must contain one name per factor")
        if not all(name and isinstance(name, str) for name in names):
            raise ValueError("factor names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")
        if not self.dependence_assumption:
            raise ValueError("dependence_assumption must be non-empty")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "factor_names", names)

    @property
    def scenario_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def factor_count(self) -> int:
        return int(self.values.shape[1])

    def linear_loss_distribution(
        self,
        exposures: ArrayLike,
        *,
        intercept: float = 0.0,
        probabilities: ArrayLike | None = None,
    ) -> LossDistribution:
        """Map factor shocks into losses using a transparent linear exposure model."""

        weights = np.asarray(exposures, dtype=np.float64).reshape(-1)
        if weights.shape != (self.factor_count,) or not np.all(np.isfinite(weights)):
            raise ValueError("exposures must contain one finite value per factor")
        if not isfinite(intercept):
            raise ValueError("intercept must be finite")
        losses = np.asarray(intercept + self.values @ weights, dtype=np.float64)
        scenario_probabilities = (
            None
            if probabilities is None
            else np.asarray(probabilities, dtype=np.float64).reshape(-1)
        )
        return LossDistribution(losses, scenario_probabilities)


@dataclass(frozen=True, slots=True)
class GaussianFactorModel:
    """Correlated Gaussian factor model for research-grade scenario generation.

    The dependence assumption is deliberately explicit. This is a Gaussian
    linear-factor foundation, not a claim that financial tails are Gaussian.
    """

    factor_names: Sequence[str]
    correlation: FloatArray
    means: FloatArray | None = None
    standard_deviations: FloatArray | None = None

    def __post_init__(self) -> None:
        names = tuple(self.factor_names)
        correlation = np.ascontiguousarray(self.correlation, dtype=np.float64)
        if not names or not all(name and isinstance(name, str) for name in names):
            raise ValueError("factor_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")
        factor_count = len(names)
        if correlation.shape != (factor_count, factor_count):
            raise ValueError("correlation must be square with one row per factor")
        if not np.all(np.isfinite(correlation)):
            raise ValueError("correlation must be finite")
        if not np.allclose(correlation, correlation.T, atol=1e-12, rtol=0.0):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("correlation diagonal must equal one")
        if np.any(np.abs(correlation) > 1.0 + 1e-12):
            raise ValueError("correlations must lie in [-1, 1]")
        eigenvalues = np.linalg.eigvalsh(correlation)
        if float(np.min(eigenvalues)) < -1e-10:
            raise ValueError("correlation must be positive semidefinite")

        means = (
            np.zeros(factor_count, dtype=np.float64)
            if self.means is None
            else np.asarray(self.means, dtype=np.float64).reshape(-1)
        )
        standard_deviations = (
            np.ones(factor_count, dtype=np.float64)
            if self.standard_deviations is None
            else np.asarray(self.standard_deviations, dtype=np.float64).reshape(-1)
        )
        if means.shape != (factor_count,) or not np.all(np.isfinite(means)):
            raise ValueError("means must contain one finite value per factor")
        if (
            standard_deviations.shape != (factor_count,)
            or not np.all(np.isfinite(standard_deviations))
            or np.any(standard_deviations <= 0)
        ):
            raise ValueError(
                "standard_deviations must contain one finite positive value per factor"
            )

        correlation.setflags(write=False)
        means.setflags(write=False)
        standard_deviations.setflags(write=False)
        object.__setattr__(self, "factor_names", names)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "standard_deviations", standard_deviations)

    @property
    def factor_count(self) -> int:
        return len(self.factor_names)

    def simulate(
        self,
        scenario_count: int,
        *,
        seed: int | None = None,
        antithetic: bool = False,
    ) -> FactorScenarios:
        """Generate correlated factor shocks without nested Python loops."""

        if scenario_count < 1:
            raise ValueError("scenario_count must be positive")
        eigenvalues, eigenvectors = np.linalg.eigh(self.correlation)
        square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        generator = np.random.default_rng(seed)
        draw_count = (scenario_count + 1) // 2 if antithetic else scenario_count
        independent = generator.standard_normal((draw_count, self.factor_count))
        if antithetic:
            independent = np.concatenate((independent, -independent), axis=0)[:scenario_count]
        correlated = independent @ square_root.T
        assert self.means is not None
        assert self.standard_deviations is not None
        values = self.means + correlated * self.standard_deviations
        return FactorScenarios(
            values=np.asarray(values, dtype=np.float64),
            factor_names=self.factor_names,
            dependence_assumption="Gaussian correlation with linear factor mapping",
        )


__all__ = ["FactorScenarios", "GaussianFactorModel"]
