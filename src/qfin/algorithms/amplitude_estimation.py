"""Backend-independent maximum-likelihood amplitude estimation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import asin, cos, log, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import beta

from qfin._validation import require_integer, require_integer_sequence
from qfin.exceptions import QFinTypeError, QFinValidationError

_THETA_MAX = pi / 2.0
_WILKS_95_LIKELIHOOD_RATIO = 3.841_458_820_694_124
_WILKS_95_LOG_LIKELIHOOD_DROP = _WILKS_95_LIKELIHOOD_RATIO / 2.0


def _validated_schedule(
    schedule: Sequence[int],
    shots: int,
) -> tuple[tuple[int, ...], int]:
    """Validate one MLAE schedule and its common shot count."""

    powers = require_integer_sequence(schedule, "schedule", minimum=0)
    if not powers:
        raise QFinValidationError("schedule must contain unique, non-negative powers")
    if len(set(powers)) != len(powers):
        raise QFinValidationError("schedule must contain unique, non-negative powers")
    shot_count = require_integer(shots, "shots", minimum=1)
    return powers, shot_count


@dataclass(frozen=True, slots=True)
class CircuitObservation:
    """Binomial observation from one Grover power."""

    power: int
    successes: int
    shots: int

    def __post_init__(self) -> None:
        power = require_integer(self.power, "power", minimum=0)
        shots = require_integer(self.shots, "shots", minimum=1)
        successes = require_integer(self.successes, "successes", minimum=0)
        if successes > shots:
            raise QFinValidationError("successes must lie between zero and shots")
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "successes", successes)

    @property
    def observed_probability(self) -> float:
        return self.successes / self.shots


@dataclass(frozen=True, slots=True)
class AmplitudeEstimate:
    """MLAE point estimate and likelihood-ratio uncertainty metadata.

    ``lower_95`` and ``upper_95`` are the hull of the guarded 95% confidence
    region. The full region is retained because an ambiguous Grover schedule can
    produce multiple disjoint likelihood modes. The raw likelihood-ratio region
    and local Fisher interval are diagnostic metadata.
    """

    amplitude: float
    lower_95: float
    upper_95: float
    theta: float
    log_likelihood: float
    observations: tuple[CircuitObservation, ...]
    confidence_regions_95: tuple[tuple[float, float], ...] = ()
    likelihood_ratio_regions_95: tuple[tuple[float, float], ...] = ()
    confidence_method: str = "likelihood_ratio_with_exact_binomial_guard"
    likelihood_ratio_cutoff: float = _WILKS_95_LIKELIHOOD_RATIO
    fisher_information: float | None = None
    fisher_interval_95: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        regions = self.confidence_regions_95 or ((self.lower_95, self.upper_95),)
        likelihood_regions = self.likelihood_ratio_regions_95 or regions
        return {
            "amplitude": self.amplitude,
            "confidence_interval_95": [self.lower_95, self.upper_95],
            "confidence_regions_95": [list(region) for region in regions],
            "likelihood_ratio_regions_95": [list(region) for region in likelihood_regions],
            "confidence_method": self.confidence_method,
            "likelihood_ratio_cutoff": self.likelihood_ratio_cutoff,
            "fisher_information": self.fisher_information,
            "fisher_interval_95": (
                None if self.fisher_interval_95 is None else list(self.fisher_interval_95)
            ),
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
    with np.errstate(divide="ignore"):
        for observation in observations:
            angle = (2 * observation.power + 1) * theta
            if observation.successes:
                sine = np.where(theta == 0.0, 0.0, np.abs(np.sin(angle)))
                likelihood += 2 * observation.successes * np.log(sine)
            failures = observation.shots - observation.successes
            if failures:
                cosine = np.where(theta == _THETA_MAX, 0.0, np.abs(np.cos(angle)))
                likelihood += 2 * failures * np.log(cosine)
    return likelihood


def _scalar_log_likelihood(
    theta: float,
    observations: Sequence[CircuitObservation],
) -> float:
    likelihood = 0.0
    for observation in observations:
        angle = (2 * observation.power + 1) * theta
        if observation.successes:
            sine = 0.0 if theta == 0.0 else abs(sin(angle))
            if sine == 0.0:
                return -float("inf")
            likelihood += 2 * observation.successes * log(sine)
        failures = observation.shots - observation.successes
        if failures:
            cosine = 0.0 if theta == _THETA_MAX else abs(cos(angle))
            if cosine == 0.0:
                return -float("inf")
            likelihood += 2 * failures * log(cosine)
    return likelihood


def _coarse_to_fine_maxima(
    observations: Sequence[CircuitObservation],
    maximum_grid_size: int,
) -> tuple[
    float,
    float,
    tuple[tuple[float, float], ...],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    # Between consecutive sine/cosine zeros, each nonconstant term has negative
    # second derivative: -2*n*w**2*csc(w*t)**2 or -2*n*w**2*sec(w*t)**2.
    # Thus every interval is concave and has at most one interior maximum.
    # Searching all intervals avoids the missed-mode risk of a uniform grid.
    frequencies = tuple(2 * item.power + 1 for item in observations)
    if sum(frequencies) + 1 > maximum_grid_size:
        raise QFinValidationError(
            "grid_size is too small to resolve every Grover likelihood interval; "
            "increase grid_size or reduce the Grover powers"
        )
    boundaries = np.unique(
        np.concatenate(
            [
                np.arange(frequency + 1, dtype=np.float64) * (_THETA_MAX / frequency)
                for frequency in frequencies
            ]
        )
    )
    # Coincident rational breakpoints can differ by one rounding bit.
    boundaries = boundaries[np.r_[True, np.diff(boundaries) > 4 * np.finfo(float).eps]]
    boundaries[0], boundaries[-1] = 0.0, _THETA_MAX
    likelihood = _log_likelihood(boundaries, observations)
    candidates = [
        (float(theta), float(value))
        for theta, value in zip(boundaries, likelihood, strict=True)
        if np.isfinite(value)
    ]
    for lower, upper in pairwise(boundaries):
        result = minimize_scalar(
            lambda value: -_scalar_log_likelihood(value, observations),
            bounds=(float(lower), float(upper)),
            method="bounded",
            options={"xatol": 1e-14},
        )
        if not result.success:
            raise QFinValidationError("MLAE likelihood refinement failed to converge")
        refined_theta = float(result.x)
        candidates.append((refined_theta, _scalar_log_likelihood(refined_theta, observations)))
    maxima = tuple(candidates)
    theta, log_likelihood = max(maxima, key=lambda item: (item[1], -item[0]))
    return theta, log_likelihood, maxima, boundaries, likelihood


def _dense_reference_maximum_likelihood(
    observations: Sequence[CircuitObservation],
    *,
    grid_size: int = 131_073,
) -> tuple[float, float]:
    """Return the legacy dense-grid estimate for numerical validation."""

    resolved_grid_size = require_integer(grid_size, "grid_size", minimum=1_001)
    theta_grid = np.linspace(0.0, _THETA_MAX, resolved_grid_size, dtype=np.float64)
    likelihood = _log_likelihood(theta_grid, observations)
    best_index = int(np.argmax(likelihood))
    return float(theta_grid[best_index]), float(likelihood[best_index])


def _crossing(
    function: Callable[[float], float],
    lower: float,
    upper: float,
) -> float:
    lower_value = function(lower)
    if lower_value == 0.0:
        return lower
    upper_value = function(upper)
    if upper_value == 0.0:
        return upper
    return float(brentq(function, lower, upper, xtol=1e-14, rtol=1e-14))


def _likelihood_ratio_regions(
    observations: Sequence[CircuitObservation],
    theta_grid: NDArray[np.float64],
    maxima: Sequence[tuple[float, float]],
    maximum_log_likelihood: float,
) -> tuple[tuple[float, float], ...]:
    threshold = maximum_log_likelihood - _WILKS_95_LOG_LIKELIHOOD_DROP
    augmented_theta = np.unique(
        np.concatenate(
            (
                theta_grid,
                np.asarray([item[0] for item in maxima], dtype=np.float64),
            )
        )
    )
    augmented_likelihood = _log_likelihood(augmented_theta, observations)
    difference = augmented_likelihood - threshold

    def threshold_difference(theta: float) -> float:
        return _scalar_log_likelihood(theta, observations) - threshold

    regions: list[tuple[float, float]] = []
    start = float(augmented_theta[0]) if difference[0] >= 0.0 else None
    for index in range(augmented_theta.size - 1):
        lower = float(augmented_theta[index])
        upper = float(augmented_theta[index + 1])
        lower_inside = difference[index] >= 0.0
        upper_inside = difference[index + 1] >= 0.0
        if not lower_inside and upper_inside:
            start = _crossing(threshold_difference, lower, upper)
        elif lower_inside and not upper_inside:
            end = _crossing(threshold_difference, lower, upper)
            regions.append((lower if start is None else start, end))
            start = None
    if start is not None:
        regions.append((start, float(augmented_theta[-1])))

    if not regions:
        best_theta, _ = max(maxima, key=lambda item: (item[1], -item[0]))
        regions.append((best_theta, best_theta))

    return tuple(regions)


def _merge_regions(
    regions: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    if not regions:
        return ()
    ordered = sorted(regions)
    merged = [ordered[0]]
    for lower, upper in ordered[1:]:
        previous_lower, previous_upper = merged[-1]
        tolerance = 1e-14 * max(1.0, abs(previous_upper), abs(lower))
        if lower <= previous_upper + tolerance:
            merged[-1] = (previous_lower, max(previous_upper, upper))
        else:
            merged.append((lower, upper))
    return tuple(merged)


def _intersect_regions(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    intersections: list[tuple[float, float]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        lower = max(left[left_index][0], right[right_index][0])
        upper = min(left[left_index][1], right[right_index][1])
        if lower <= upper:
            intersections.append((lower, upper))
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return _merge_regions(intersections)


def _probability_constraint_regions(
    power: int,
    lower_probability: float,
    upper_probability: float,
) -> tuple[tuple[float, float], ...]:
    frequency = 2 * power + 1
    lower_angle = asin(sqrt(lower_probability))
    upper_angle = asin(sqrt(upper_probability))
    regions: list[tuple[float, float]] = []
    for half_wave in range(frequency):
        period = half_wave // 2
        if half_wave % 2 == 0:
            lower = (period * pi + lower_angle) / frequency
            upper = (period * pi + upper_angle) / frequency
        else:
            upper_turn = (period + 1) * pi
            lower = (upper_turn - upper_angle) / frequency
            upper = (upper_turn - lower_angle) / frequency
        regions.append((max(0.0, lower), min(_THETA_MAX, upper)))
    return _merge_regions(regions)


def _exact_binomial_confidence_regions(
    observations: Sequence[CircuitObservation],
) -> tuple[tuple[float, float], ...]:
    """Return a finite-sample simultaneous 95% confidence set in theta."""

    family_tail_probability = 0.05 / (2.0 * len(observations))
    regions: tuple[tuple[float, float], ...] = ((0.0, _THETA_MAX),)
    for observation in observations:
        if observation.successes == 0:
            lower_probability = 0.0
        else:
            lower_probability = float(
                beta.ppf(
                    family_tail_probability,
                    observation.successes,
                    observation.shots - observation.successes + 1,
                )
            )
        if observation.successes == observation.shots:
            upper_probability = 1.0
        else:
            upper_probability = float(
                beta.ppf(
                    1.0 - family_tail_probability,
                    observation.successes + 1,
                    observation.shots - observation.successes,
                )
            )
        constraint = _probability_constraint_regions(
            observation.power,
            lower_probability,
            upper_probability,
        )
        regions = _intersect_regions(regions, constraint)
        if not regions:
            break
    return regions


def _amplitude_regions(
    regions: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    return tuple((sin(lower) ** 2, sin(upper) ** 2) for lower, upper in regions)


def maximum_likelihood_amplitude_estimate(
    observations: Sequence[CircuitObservation],
    *,
    grid_size: int = 131_073,
) -> AmplitudeEstimate:
    """Fit an amplitude and its 95% likelihood-ratio confidence region.

    For Grover power ``k``, the objective-qubit success probability is
    ``sin²((2k+1) theta)`` and the desired amplitude is ``sin²(theta)``.
    The coarse partition uses every sine/cosine zero of the Grover likelihood.
    Each resulting interval is concave; bounded local searches refine every
    candidate and select the global maximum. ``grid_size`` bounds the number
    of partition points; an undersized budget is explicitly rejected.

    The reported 95% region contains the profile likelihood-ratio region using
    the one-parameter Wilks cutoff. To avoid relying on that asymptotic cutoff at
    a boundary or with few shots, it is unioned with a simultaneous exact
    Clopper-Pearson set for the observed circuit probabilities. The result can be
    disjoint for an ambiguous Grover schedule. ``lower_95`` and ``upper_95``
    retain the backward-compatible hull. The local Fisher interval is diagnostic
    metadata only.
    """

    items = tuple(observations)
    if not items:
        raise QFinValidationError("at least one circuit observation is required")
    resolved_grid_size = require_integer(grid_size, "grid_size", minimum=1_001)
    if any(not isinstance(item, CircuitObservation) for item in items):
        raise QFinTypeError("observations must contain CircuitObservation objects")
    if len({item.power for item in items}) != len(items):
        raise QFinValidationError("each Grover power may appear only once")

    theta, log_likelihood, maxima, theta_grid, _ = _coarse_to_fine_maxima(
        items,
        resolved_grid_size,
    )
    amplitude = sin(theta) ** 2
    likelihood_theta_regions = _likelihood_ratio_regions(
        items,
        theta_grid,
        maxima,
        log_likelihood,
    )
    exact_theta_regions = _exact_binomial_confidence_regions(items)
    theta_regions = _merge_regions((*likelihood_theta_regions, *exact_theta_regions))
    regions = _amplitude_regions(theta_regions)
    likelihood_regions = _amplitude_regions(likelihood_theta_regions)

    fisher_information = sum(4.0 * item.shots * (2 * item.power + 1) ** 2 for item in items)
    theta_standard_error = 1.0 / sqrt(fisher_information)
    theta_lower = max(0.0, theta - 1.96 * theta_standard_error)
    theta_upper = min(_THETA_MAX, theta + 1.96 * theta_standard_error)
    fisher_interval = (sin(theta_lower) ** 2, sin(theta_upper) ** 2)

    return AmplitudeEstimate(
        amplitude=amplitude,
        lower_95=regions[0][0],
        upper_95=regions[-1][1],
        theta=theta,
        log_likelihood=log_likelihood,
        observations=items,
        confidence_regions_95=regions,
        likelihood_ratio_regions_95=likelihood_regions,
        fisher_information=fisher_information,
        fisher_interval_95=fisher_interval,
    )


def direct_sampling_standard_error(amplitude: float, shots: int) -> float:
    """Return the Bernoulli standard error for comparison with MLAE."""
    if not 0 <= amplitude <= 1:
        raise QFinValidationError("amplitude must lie in [0, 1]")
    shot_count = require_integer(shots, "shots", minimum=1)
    return sqrt(amplitude * (1 - amplitude) / shot_count)


def amplitude_to_theta(amplitude: float) -> float:
    """Convert a success amplitude in ``[0, 1]`` to its Grover angle."""
    if not 0 <= amplitude <= 1:
        raise QFinValidationError("amplitude must lie in [0, 1]")
    return asin(sqrt(amplitude))
