"""Normalized quantum objectives over an encoded financial distribution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from qfin.representation.encoding import DistributionEncoding

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class QuantumObjectiveEncoding:
    """Pair a probability state with a normalized objective-qubit rotation.

    If ``a`` is the measured objective-qubit success probability, the
    corresponding financial value is ``financial_offset + financial_scale*a``.
    """

    distribution: DistributionEncoding
    normalized_values: FloatArray
    financial_scale: float
    financial_offset: float
    label: str
    threshold: float | None = None
    inclusive: bool | None = None

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.normalized_values, dtype=np.float64).reshape(-1)
        if values.shape != self.distribution.probabilities.shape:
            raise ValueError("normalized_values must match the distribution grid")
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("normalized_values must be finite and lie in [0, 1]")
        if not isfinite(self.financial_scale) or self.financial_scale < 0:
            raise ValueError("financial_scale must be finite and non-negative")
        if not isfinite(self.financial_offset):
            raise ValueError("financial_offset must be finite")
        if not self.label:
            raise ValueError("label must be non-empty")
        if self.threshold is not None and not isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "normalized_values", values)

    @property
    def exact_amplitude(self) -> float:
        return float(np.dot(self.distribution.probabilities, self.normalized_values))

    @property
    def exact_value(self) -> float:
        return self.value_from_amplitude(self.exact_amplitude)

    def value_from_amplitude(self, amplitude: float) -> float:
        if not isfinite(amplitude) or not 0 <= amplitude <= 1:
            raise ValueError("amplitude must be finite and lie in [0, 1]")
        return self.financial_offset + self.financial_scale * amplitude

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "inclusive": self.inclusive,
            "financial_scale": self.financial_scale,
            "financial_offset": self.financial_offset,
            "exact_amplitude": self.exact_amplitude,
            "exact_value": self.exact_value,
            "grid_points": self.distribution.grid_points,
            "state_preparation_method": self.distribution.state_preparation_method,
            "objective_rotation_method": "multiplexed_ry",
        }


def cdf_objective(
    distribution: DistributionEncoding,
    threshold: float,
) -> QuantumObjectiveEncoding:
    """Encode ``P(loss <= threshold)`` as an objective-qubit amplitude."""

    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    values = np.asarray(distribution.grid <= threshold, dtype=np.float64)
    return QuantumObjectiveEncoding(
        distribution=distribution,
        normalized_values=values,
        financial_scale=1.0,
        financial_offset=0.0,
        label="cdf_probability",
        threshold=threshold,
        inclusive=True,
    )


def tail_probability_objective(
    distribution: DistributionEncoding,
    threshold: float,
    *,
    inclusive: bool = False,
) -> QuantumObjectiveEncoding:
    """Encode a strict or inclusive upper-tail probability."""

    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    mask = distribution.grid >= threshold if inclusive else distribution.grid > threshold
    return QuantumObjectiveEncoding(
        distribution=distribution,
        normalized_values=np.asarray(mask, dtype=np.float64),
        financial_scale=1.0,
        financial_offset=0.0,
        label="tail_probability",
        threshold=threshold,
        inclusive=inclusive,
    )


def tail_excess_objective(
    distribution: DistributionEncoding,
    threshold: float,
) -> QuantumObjectiveEncoding:
    """Encode ``E[max(loss-threshold, 0)]`` with an explicit financial scale."""

    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    excess = np.maximum(distribution.grid - threshold, 0.0)
    scale = float(np.max(excess, initial=0.0))
    normalized = np.zeros_like(excess) if scale == 0.0 else excess / scale
    return QuantumObjectiveEncoding(
        distribution=distribution,
        normalized_values=np.asarray(normalized, dtype=np.float64),
        financial_scale=scale,
        financial_offset=0.0,
        label="tail_excess_expectation",
        threshold=threshold,
        inclusive=False,
    )


__all__ = [
    "QuantumObjectiveEncoding",
    "cdf_objective",
    "tail_excess_objective",
    "tail_probability_objective",
]
