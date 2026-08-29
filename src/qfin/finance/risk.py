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
            probabilities = np.ascontiguousarray(self.probabilities, dtype=np.float64).reshape(-1)
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


@dataclass(frozen=True, slots=True)
class VaR:
    """Compiler-facing value-at-risk problem over a finite loss distribution."""

    distribution: LossDistribution
    confidence: float = 0.995

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class TailProbability:
    """Probability that a finite loss crosses a specified threshold."""

    distribution: LossDistribution
    threshold: float
    inclusive: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite")


@dataclass(frozen=True, slots=True)
class TailProbabilitySummary:
    """Classical probability of crossing a loss threshold."""

    threshold: float
    inclusive: bool
    probability: float


@dataclass(frozen=True, slots=True)
class RiskConfidenceInterval:
    """Reproducible percentile-bootstrap intervals for empirical risk statistics."""

    confidence: float
    interval_level: float
    resamples: int
    sample_size: int
    value_at_risk: tuple[float, float]
    expected_shortfall: tuple[float, float]
    seed: int | None
    method: str = "weighted_empirical_percentile_bootstrap"

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "interval_level": self.interval_level,
            "resamples": self.resamples,
            "sample_size": self.sample_size,
            "value_at_risk": list(self.value_at_risk),
            "expected_shortfall": list(self.expected_shortfall),
            "seed": self.seed,
            "method": self.method,
            "caveat": (
                "Percentile bootstrap over the supplied empirical distribution; "
                "it does not include model, parameter, or scenario-design uncertainty."
            ),
        }


def evaluate_tail_probability(problem: TailProbability) -> TailProbabilitySummary:
    """Evaluate a finite-distribution threshold event exactly."""

    probabilities = problem.distribution.probabilities
    assert probabilities is not None
    mask = (
        problem.distribution.losses >= problem.threshold
        if problem.inclusive
        else problem.distribution.losses > problem.threshold
    )
    return TailProbabilitySummary(
        threshold=problem.threshold,
        inclusive=problem.inclusive,
        probability=float(np.sum(probabilities[mask])),
    )


def bootstrap_risk_interval(
    distribution: LossDistribution,
    *,
    confidence: float = 0.995,
    interval_level: float = 0.95,
    resamples: int = 1_000,
    sample_size: int | None = None,
    seed: int | None = 0,
) -> RiskConfidenceInterval:
    """Estimate sampling intervals for empirical VaR and expected shortfall.

    Sampling uses the supplied scenario probabilities and a local NumPy random
    generator. The default seed makes validation and examples reproducible.
    """

    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isfinite(interval_level) or not 0 < interval_level < 1:
        raise ValueError("interval_level must lie strictly between zero and one")
    if resamples < 2:
        raise ValueError("resamples must be at least two")
    resolved_sample_size = distribution.losses.size if sample_size is None else sample_size
    if resolved_sample_size < 1:
        raise ValueError("sample_size must be positive")
    probabilities = distribution.probabilities
    assert probabilities is not None
    generator = np.random.default_rng(seed)
    var_samples = np.empty(resamples, dtype=np.float64)
    cvar_samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw = generator.choice(
            distribution.losses,
            size=resolved_sample_size,
            replace=True,
            p=probabilities,
        )
        summary = _numpy_risk(LossDistribution(draw), confidence)
        var_samples[index] = summary["value_at_risk"]
        cvar_samples[index] = summary["expected_shortfall"]
    lower_quantile = 0.5 * (1.0 - interval_level)
    upper_quantile = 1.0 - lower_quantile
    var_bounds = np.quantile(var_samples, [lower_quantile, upper_quantile])
    cvar_bounds = np.quantile(cvar_samples, [lower_quantile, upper_quantile])
    return RiskConfidenceInterval(
        confidence=confidence,
        interval_level=interval_level,
        resamples=resamples,
        sample_size=resolved_sample_size,
        value_at_risk=(float(var_bounds[0]), float(var_bounds[1])),
        expected_shortfall=(float(cvar_bounds[0]), float(cvar_bounds[1])),
        seed=seed,
    )


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


__all__ = [
    "CVaR",
    "LossDistribution",
    "RiskConfidenceInterval",
    "RiskSummary",
    "TailProbability",
    "TailProbabilitySummary",
    "VaR",
    "aggregate_risk",
    "bootstrap_risk_interval",
    "evaluate_tail_probability",
]
