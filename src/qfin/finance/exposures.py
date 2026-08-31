"""Sparse multivariate exposure models on factorized financial representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

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
            latent if self.encoding.transform is None else self.encoding.transform.apply(latent)
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


@dataclass(frozen=True, slots=True)
class FactorVaR:
    """Value-at-risk problem over a factorized loss model."""

    model: FactorizedLossModel
    confidence: float = 0.995

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class FactorCVaR:
    """Expected-shortfall problem over a factorized loss model."""

    model: FactorizedLossModel
    confidence: float = 0.995

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")


FactorRiskProblem = FactorVaR | FactorCVaR


@dataclass(frozen=True, slots=True)
class FactorRiskSummary:
    """Memory-bounded classical VaR/CVaR reference on a factor grid."""

    problem_kind: Literal["value_at_risk", "conditional_value_at_risk"]
    confidence: float
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    value_at_risk: float
    expected_shortfall: float
    evaluated_points: int
    streamed_point_visits: int
    chunks: int
    cdf_evaluations: int
    joint_table_materialized: bool = False

    @property
    def var(self) -> float:
        return self.value_at_risk

    @property
    def cvar(self) -> float:
        return self.expected_shortfall

    @property
    def value(self) -> float:
        return self.var if self.problem_kind == "value_at_risk" else self.cvar

    def to_dict(self) -> dict[str, object]:
        return {
            "problem_kind": self.problem_kind,
            "confidence": self.confidence,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "value_at_risk": self.value_at_risk,
            "expected_shortfall": self.expected_shortfall,
            "evaluated_points": self.evaluated_points,
            "streamed_point_visits": self.streamed_point_visits,
            "chunks": self.chunks,
            "cdf_evaluations": self.cdf_evaluations,
            "joint_table_materialized": self.joint_table_materialized,
        }


_SIGN_MASK = 1 << 63
_UINT64_MASK = (1 << 64) - 1


def _float_order_key(value: float) -> int:
    """Map a finite IEEE-754 double to a monotonically ordered integer."""

    bits = int(np.asarray(value, dtype=np.float64).view(np.uint64))
    return (~bits & _UINT64_MASK) if bits & _SIGN_MASK else bits ^ _SIGN_MASK


def _float_from_order_key(key: int) -> float:
    bits = (~key & _UINT64_MASK) if key < _SIGN_MASK else key ^ _SIGN_MASK
    return float(np.asarray(bits, dtype=np.uint64).view(np.float64))


def evaluate_factor_risk(
    problem: FactorRiskProblem,
    *,
    chunk_size: int = 65_536,
    max_points: int = 1_048_576,
) -> FactorRiskSummary:
    """Evaluate exact encoded-grid VaR/CVaR without a joint loss table.

    The weighted quantile is selected by a 64-bit monotone search over the
    IEEE-754 value domain. This deliberately trades repeated streamed passes
    for bounded memory and is a correctness oracle, not the fast execution path.
    """

    if chunk_size < 1 or max_points < 1:
        raise ValueError("chunk_size and max_points must be positive")
    points = problem.model.joint_grid_points
    if points > max_points:
        raise ValueError(
            f"factorized validation requires {points} streamed points, "
            f"above max_points={max_points}"
        )

    total_mass = 0.0
    weighted_sum = 0.0
    weighted_square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    chunks_per_pass = 0
    for start in range(0, points, chunk_size):
        stop = min(start + chunk_size, points)
        _, losses, weights = problem.model.chunk(start, stop)
        total_mass += float(np.sum(weights))
        weighted_sum += float(np.dot(losses, weights))
        weighted_square_sum += float(np.dot(losses * losses, weights))
        minimum = min(minimum, float(np.min(losses)))
        maximum = max(maximum, float(np.max(losses)))
        chunks_per_pass += 1
    if not isfinite(total_mass) or total_mass <= 0:
        raise ValueError("factorized loss model must have positive finite probability mass")

    target_mass = problem.confidence * total_mass
    cdf_evaluations = 0

    def cdf(threshold: float) -> float:
        nonlocal cdf_evaluations
        mass = 0.0
        for start in range(0, points, chunk_size):
            stop = min(start + chunk_size, points)
            _, losses, weights = problem.model.chunk(start, stop)
            mass += float(np.sum(weights[losses <= threshold]))
        cdf_evaluations += 1
        return mass

    if cdf(minimum) >= target_mass:
        value_at_risk = minimum
    else:
        lower_key = _float_order_key(minimum)
        upper_key = _float_order_key(maximum)
        while lower_key + 1 < upper_key:
            middle_key = (lower_key + upper_key) // 2
            if cdf(_float_from_order_key(middle_key)) >= target_mass:
                upper_key = middle_key
            else:
                lower_key = middle_key
        value_at_risk = _float_from_order_key(upper_key)

    weighted_excess = 0.0
    for start in range(0, points, chunk_size):
        stop = min(start + chunk_size, points)
        _, losses, weights = problem.model.chunk(start, stop)
        weighted_excess += float(np.dot(np.maximum(losses - value_at_risk, 0.0), weights))

    mean = weighted_sum / total_mass
    variance = max(weighted_square_sum / total_mass - mean * mean, 0.0)
    expected_shortfall = value_at_risk + (weighted_excess / total_mass / (1.0 - problem.confidence))
    passes = 2 + cdf_evaluations
    kind: Literal["value_at_risk", "conditional_value_at_risk"] = (
        "value_at_risk" if isinstance(problem, FactorVaR) else "conditional_value_at_risk"
    )
    return FactorRiskSummary(
        problem_kind=kind,
        confidence=problem.confidence,
        mean=mean,
        standard_deviation=float(np.sqrt(variance)),
        minimum=minimum,
        maximum=maximum,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        evaluated_points=points,
        streamed_point_visits=passes * points,
        chunks=passes * chunks_per_pass,
        cdf_evaluations=cdf_evaluations,
    )


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
    "FactorCVaR",
    "FactorRiskProblem",
    "FactorRiskSummary",
    "FactorTailProbability",
    "FactorTailProbabilitySummary",
    "FactorVaR",
    "FactorizedLossModel",
    "HingeExposure",
    "SparseExposureObjective",
    "evaluate_factor_risk",
    "evaluate_factor_tail_probability",
]
