"""Classical loss distributions and tail-risk aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from qfin import _native
from qfin.finance.distributions import EmpiricalDistribution
from qfin.finance.fixed_income import Engine

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LossDistribution:
    """Finite losses and normalized scenario probabilities."""

    losses: FloatArray
    probabilities: FloatArray | None = None

    def __post_init__(self) -> None:
        losses = np.ascontiguousarray(self.losses, dtype=np.float64).reshape(-1)
        if losses.size == 0 or not np.all(np.isfinite(losses)):
            raise ValueError("losses must contain at least one finite value")
        if self.probabilities is None:
            probabilities = np.full(losses.size, 1.0 / losses.size, dtype=np.float64)
        else:
            probabilities = np.ascontiguousarray(
                self.probabilities, dtype=np.float64
            ).reshape(-1)
            if probabilities.shape != losses.shape:
                raise ValueError("probabilities must have the same shape as losses")
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
                raise ValueError("probabilities must be finite and non-negative")
            scale = float(np.max(probabilities, initial=0.0))
            if scale <= 0:
                raise ValueError("probabilities must have positive total mass")
            scaled = probabilities / scale
            probabilities = scaled / float(np.sum(scaled))
        losses.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "probabilities", probabilities)

    def as_empirical(self) -> EmpiricalDistribution:
        """Return the representation layer's existing empirical-distribution type."""

        assert self.probabilities is not None
        return EmpiricalDistribution(self.losses, self.probabilities)


@dataclass(frozen=True, slots=True)
class RiskSummary:
    """Weighted distribution and tail-risk statistics."""

    confidence: float
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    value_at_risk: float
    expected_shortfall: float
    engine: Literal["numpy", "native"]

    @property
    def var(self) -> float:
        return self.value_at_risk

    @property
    def cvar(self) -> float:
        return self.expected_shortfall


@dataclass(frozen=True, slots=True)
class CVaR:
    """Compiler-facing expected-shortfall problem over a finite loss distribution."""

    distribution: LossDistribution
    confidence: float = 0.995

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")


def _numpy_risk(distribution: LossDistribution, confidence: float) -> dict[str, float]:
    assert distribution.probabilities is not None
    order = np.argsort(distribution.losses, kind="stable")
    losses = distribution.losses[order]
    probabilities = distribution.probabilities[order]
    cumulative = np.cumsum(probabilities)
    index = min(int(np.searchsorted(cumulative, confidence, side="left")), losses.size - 1)
    previous = np.concatenate((np.array([0.0]), cumulative[:-1]))
    overlap = np.maximum(cumulative - np.maximum(previous, confidence), 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.dot(losses, probabilities))
        variance = float(np.dot((losses - mean) ** 2, probabilities))
    if not (isfinite(mean) and isfinite(variance)):
        raise ValueError("weighted loss moments exceed the finite double range")
    expected_shortfall = float(np.dot(losses, overlap) / (1.0 - confidence))
    if not isfinite(expected_shortfall):
        raise ValueError("expected shortfall exceeds the finite double range")
    return {
        "mean": mean,
        "standard_deviation": float(np.sqrt(max(0.0, variance))),
        "minimum": float(losses[0]),
        "maximum": float(losses[-1]),
        "value_at_risk": float(losses[index]),
        "expected_shortfall": expected_shortfall,
    }


def aggregate_risk(
    distribution: LossDistribution,
    *,
    confidence: float = 0.995,
    engine: Engine = "auto",
) -> RiskSummary:
    """Compute weighted VaR and coherent discrete expected shortfall."""

    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        # Measured sort-heavy tail aggregation has no stable native crossover yet.
        # Keep automatic execution on NumPy; native remains an explicit profiling
        # and parity path until repeatable evidence supports a threshold.
        selected = "numpy"
    assert distribution.probabilities is not None
    if selected == "native":
        raw = cast(
            dict[str, object],
            _native.require().aggregate_tail_risk(
                distribution.losses, distribution.probabilities, confidence
            ),
        )
        value_names = (
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "value_at_risk",
            "expected_shortfall",
        )
        values = {name: float(cast(float, raw[name])) for name in value_names}
    else:
        values = _numpy_risk(distribution, confidence)
    return RiskSummary(
        confidence=confidence,
        mean=values["mean"],
        standard_deviation=values["standard_deviation"],
        minimum=values["minimum"],
        maximum=values["maximum"],
        value_at_risk=values["value_at_risk"],
        expected_shortfall=values["expected_shortfall"],
        engine=selected,
    )


__all__ = ["CVaR", "LossDistribution", "RiskSummary", "aggregate_risk"]
