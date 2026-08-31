"""Sparse multivariate exposure models on factorized financial representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from qfin.representation.factorized import FactorizedDistributionEncoding

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HingeExposure:
    """One piecewise-linear term ``slope * max(factor - threshold, 0)``."""

    factor: str
    threshold: float
    slope: float

    def __post_init__(self) -> None:
        if not self.factor.strip():
            raise ValueError("factor must not be empty")
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not isfinite(self.slope) or self.slope == 0:
            raise ValueError("slope must be finite and non-zero")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "kind": "positive_part",
            "factor": self.factor,
            "threshold": self.threshold,
            "slope": self.slope,
        }


@dataclass(frozen=True, slots=True)
class SparseExposureObjective:
    """Constant, sparse linear/quadratic, and univariate hinge exposures.

    A quadratic mapping entry ``(left, right): coefficient`` contributes
    ``coefficient * left * right`` exactly once. Piecewise terms are explicit
    positive parts rather than arbitrary Python callables, which lets the
    representation layer construct a reversible arithmetic plan.
    """

    constant: float = 0.0
    linear: Mapping[str, float] = field(default_factory=dict)
    quadratic: Mapping[tuple[str, str], float] = field(default_factory=dict)
    piecewise: Sequence[HingeExposure] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.constant):
            raise ValueError("constant must be finite")

        linear: dict[str, float] = {}
        for name, coefficient in self.linear.items():
            value = float(coefficient)
            if not name.strip() or not isfinite(value):
                raise ValueError("linear exposures require non-empty names and finite values")
            if value != 0:
                linear[name] = value

        quadratic: dict[tuple[str, str], float] = {}
        for pair, coefficient in self.quadratic.items():
            if len(pair) != 2 or not pair[0].strip() or not pair[1].strip():
                raise ValueError("quadratic keys must contain two non-empty factor names")
            value = float(coefficient)
            if not isfinite(value):
                raise ValueError("quadratic coefficients must be finite")
            left, right = sorted((pair[0], pair[1]))
            key = (left, right)
            quadratic[key] = quadratic.get(key, 0.0) + value
        quadratic = {key: value for key, value in quadratic.items() if value != 0}

        piecewise = tuple(self.piecewise)
        if not all(isinstance(term, HingeExposure) for term in piecewise):
            raise TypeError("piecewise entries must be HingeExposure objects")

        object.__setattr__(self, "linear", MappingProxyType(linear))
        object.__setattr__(self, "quadratic", MappingProxyType(quadratic))
        object.__setattr__(self, "piecewise", piecewise)

    @property
    def referenced_factors(self) -> tuple[str, ...]:
        names = set(self.linear)
        for left, right in self.quadratic:
            names.add(left)
            names.add(right)
        names.update(term.factor for term in self.piecewise)
        return tuple(sorted(names))

    @property
    def polynomial_terms(self) -> int:
        return int(self.constant != 0) + len(self.linear) + len(self.quadratic)

    def evaluate(
        self,
        factor_values: ArrayLike,
        factor_names: Sequence[str],
    ) -> FloatArray:
        """Evaluate the financial objective on a scenario-by-factor array."""

        values = np.asarray(factor_values, dtype=np.float64)
        names = tuple(factor_names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("factor_values must be scenario-by-factor")
        if not np.all(np.isfinite(values)):
            raise ValueError("factor_values must be finite")
        if len(set(names)) != len(names):
            raise ValueError("factor_names must be unique")
        missing = set(self.referenced_factors) - set(names)
        if missing:
            raise ValueError(
                "objective references unavailable factors: " + ", ".join(sorted(missing))
            )

        column = {name: index for index, name in enumerate(names)}
        result = np.full(values.shape[0], self.constant, dtype=np.float64)
        for name, coefficient in self.linear.items():
            result += coefficient * values[:, column[name]]
        for (left, right), coefficient in self.quadratic.items():
            result += coefficient * values[:, column[left]] * values[:, column[right]]
        for term in self.piecewise:
            result += term.slope * np.maximum(values[:, column[term.factor]] - term.threshold, 0.0)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "constant": self.constant,
            "linear": dict(self.linear),
            "quadratic": {
                f"{left}*{right}": coefficient
                for (left, right), coefficient in self.quadratic.items()
            },
            "piecewise": [term.to_dict() for term in self.piecewise],
            "referenced_factors": list(self.referenced_factors),
        }


@dataclass(frozen=True, slots=True)
class FactorizedLossModel:
    """A sparse financial loss objective over factorized encoded registers."""

    encoding: FactorizedDistributionEncoding
    objective: SparseExposureObjective

    def __post_init__(self) -> None:
        from qfin.representation.factorized import FactorizedDistributionEncoding

        if not isinstance(self.encoding, FactorizedDistributionEncoding):
            raise TypeError("encoding must be a FactorizedDistributionEncoding")
        available = set(self.encoding.value_names)
        missing = set(self.objective.referenced_factors) - available
        if missing:
            raise ValueError(
                "objective references unavailable factors: " + ", ".join(sorted(missing))
            )

    @property
    def joint_grid_points(self) -> int:
        return self.encoding.joint_grid_points

    def chunk(
        self,
        start: int,
        stop: int,
    ) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
        """Return indices, losses, and probabilities for one bounded flat slice."""

        if not 0 <= start <= stop <= self.joint_grid_points:
            raise ValueError("chunk bounds must lie inside the joint index range")
        flat = np.arange(start, stop, dtype=np.int64)
        residual = flat.copy()
        indices = np.empty((flat.size, self.encoding.factor_count), dtype=np.int64)
        for factor_index in range(self.encoding.factor_count - 1, -1, -1):
            points = self.encoding.factors[factor_index].grid_points
            indices[:, factor_index] = residual % points
            residual //= points

        latent = np.column_stack(
            [
                factor.grid[indices[:, factor_index]]
                for factor_index, factor in enumerate(self.encoding.factors)
            ]
        )
        values = (
            latent
            if self.encoding.transform is None
            else self.encoding.transform.apply(latent)
        )
        losses = self.objective.evaluate(values, self.encoding.value_names)
        probabilities = np.ones(flat.size, dtype=np.float64)
        for factor_index, factor in enumerate(self.encoding.factors):
            probabilities *= factor.probabilities[indices[:, factor_index]]
        return indices, losses, probabilities

    def to_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding.to_dict(),
            "objective": self.objective.to_dict(),
            "joint_table_materialized": False,
        }


@dataclass(frozen=True, slots=True)
class FactorTailProbability:
    """Tail-probability problem whose loss is computed from factor registers."""

    model: FactorizedLossModel
    threshold: float
    inclusive: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite")


@dataclass(frozen=True, slots=True)
class FactorTailProbabilitySummary:
    """Streaming classical reference for a factorized tail objective."""

    probability: float
    threshold: float
    inclusive: bool
    evaluated_points: int
    chunks: int
    joint_table_materialized: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "threshold": self.threshold,
            "inclusive": self.inclusive,
            "evaluated_points": self.evaluated_points,
            "chunks": self.chunks,
            "joint_table_materialized": self.joint_table_materialized,
        }


def evaluate_factor_tail_probability(
    problem: FactorTailProbability,
    *,
    chunk_size: int = 65_536,
    max_points: int = 1_048_576,
) -> FactorTailProbabilitySummary:
    """Evaluate a factorized tail probability with bounded working memory."""

    if chunk_size < 1 or max_points < 1:
        raise ValueError("chunk_size and max_points must be positive")
    points = problem.model.joint_grid_points
    if points > max_points:
        raise ValueError(
            f"factorized validation requires {points} streamed points, "
            f"above max_points={max_points}"
        )
    probability = 0.0
    chunks = 0
    for start in range(0, points, chunk_size):
        stop = min(start + chunk_size, points)
        _, losses, weights = problem.model.chunk(start, stop)
        selected = losses >= problem.threshold if problem.inclusive else losses > problem.threshold
        probability += float(np.sum(weights[selected]))
        chunks += 1
    return FactorTailProbabilitySummary(
        probability=float(np.clip(probability, 0.0, 1.0)),
        threshold=problem.threshold,
        inclusive=problem.inclusive,
        evaluated_points=points,
        chunks=chunks,
    )


__all__ = [
    "FactorTailProbability",
    "FactorTailProbabilitySummary",
    "FactorizedLossModel",
    "HingeExposure",
    "SparseExposureObjective",
    "evaluate_factor_tail_probability",
]
