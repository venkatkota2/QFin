"""Backend-independent maximum-likelihood amplitude estimation."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, sin, sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CircuitObservation:
    """Binomial observation from one Grover power."""

    power: int
    successes: int
    shots: int

    def __post_init__(self) -> None:
        if self.power < 0:
            raise ValueError("power must be non-negative")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if not 0 <= self.successes <= self.shots:
            raise ValueError("successes must lie between zero and shots")

    @property
    def observed_probability(self) -> float:
        return self.successes / self.shots


@dataclass(frozen=True, slots=True)
class AmplitudeEstimate:
    """MLAE point estimate and local 95% confidence interval."""

    amplitude: float
    lower_95: float
    upper_95: float
    theta: float
    log_likelihood: float
    observations: tuple[CircuitObservation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "amplitude": self.amplitude,
            "confidence_interval_95": [self.lower_95, self.upper_95],
            "theta": self.theta,
            "log_likelihood": self.log_likelihood,
            "observations": [
                {
                    "power": item.power,
                    "successes": item.successes,
                    "shots": item.shots,
                    "observed_probability": item.observed_probability,
                }
                for item in self.observations
            ],
        }


def _log_likelihood(
    theta: NDArray[np.float64], observations: Sequence[CircuitObservation]
) -> NDArray[np.float64]:
    likelihood = np.zeros_like(theta)
    for observation in observations:
        probability = np.sin((2 * observation.power + 1) * theta) ** 2
        probability = np.clip(probability, 1e-15, 1 - 1e-15)
        likelihood += observation.successes * np.log(probability)
        likelihood += (observation.shots - observation.successes) * np.log1p(-probability)
    return likelihood


def maximum_likelihood_amplitude_estimate(
    observations: Sequence[CircuitObservation],
    *,
    grid_size: int = 131_073,
) -> AmplitudeEstimate:
    """Fit the amplitude from a non-adaptive MLAE schedule.

    For Grover power ``k``, the objective-qubit success probability is
    ``sin²((2k+1) theta)`` and the desired amplitude is ``sin²(theta)``.
    A dense one-dimensional search is intentional: it is deterministic and
    robust to the likelihood's multiple local maxima at MVP problem sizes.
    """

    items = tuple(observations)
    if not items:
        raise ValueError("at least one circuit observation is required")
    if grid_size < 1_001:
        raise ValueError("grid_size must be at least 1001")
    if len({item.power for item in items}) != len(items):
        raise ValueError("each Grover power may appear only once")

    theta_grid = np.linspace(0.0, np.pi / 2, grid_size, dtype=np.float64)
    likelihood = _log_likelihood(theta_grid, items)
    best_index = int(np.argmax(likelihood))
    theta = float(theta_grid[best_index])
    amplitude = sin(theta) ** 2

    fisher_information = sum(
        4.0 * item.shots * (2 * item.power + 1) ** 2 for item in items
    )
    theta_standard_error = 1.0 / sqrt(fisher_information)
    theta_lower = max(0.0, theta - 1.96 * theta_standard_error)
    theta_upper = min(np.pi / 2, theta + 1.96 * theta_standard_error)

    return AmplitudeEstimate(
        amplitude=amplitude,
        lower_95=sin(theta_lower) ** 2,
        upper_95=sin(theta_upper) ** 2,
        theta=theta,
        log_likelihood=float(likelihood[best_index]),
        observations=items,
    )


def direct_sampling_standard_error(amplitude: float, shots: int) -> float:
    """Return the Bernoulli standard error for comparison with MLAE."""
    if not 0 <= amplitude <= 1:
        raise ValueError("amplitude must lie in [0, 1]")
    if shots <= 0:
        raise ValueError("shots must be positive")
    return sqrt(amplitude * (1 - amplitude) / shots)


def amplitude_to_theta(amplitude: float) -> float:
    """Convert a success amplitude in ``[0, 1]`` to its Grover angle."""
    if not 0 <= amplitude <= 1:
        raise ValueError("amplitude must lie in [0, 1]")
    return asin(sqrt(amplitude))

