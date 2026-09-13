"""Simultaneous sampling bounds for the existing adaptive risk workflows.

No new circuits or point estimators: Bonferroni-budget the existing exact
binomial inversion before sampling, then propagate quantile-selection error.
Coverage refers to the encoded distribution under ideal independent binomial
shots conditional on earlier queries, not to encoding error or noisy hardware.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sin

import numpy as np

from qfin.algorithms import AmplitudeEstimate
from qfin.algorithms.amplitude_estimation import _exact_binomial_confidence_regions


def objective_budget(candidate_count: int, excess_objectives: int = 0) -> int:
    """Maximum binary-search queries, selected-point check, and excess queries."""

    return max(0, candidate_count - 1).bit_length() + 1 + excess_objectives


def simultaneous_amplitude_interval(
    estimate: AmplitudeEstimate, budget: int
) -> tuple[float, float]:
    """A per-objective bound with at most 0.05 / budget failure probability.

    Empty/inconsistent regions must lose precision rather than assert certainty.
    Local likelihood intervals remain diagnostic on the original estimate.
    """

    regions = _exact_binomial_confidence_regions(
        estimate.observations, failure_probability=0.05 / budget
    )
    if not regions:
        return 0.0, 1.0
    # Round the converted endpoints outward to retain boundary amplitudes.
    lower = float(np.nextafter(sin(regions[0][0]) ** 2, -np.inf))
    upper = float(np.nextafter(sin(regions[-1][1]) ** 2, np.inf))
    return max(0.0, lower), min(1.0, upper)


def quantile_indices(
    candidate_count: int,
    confidence: float,
    bounds: Sequence[tuple[int, tuple[float, float]]],
) -> tuple[int, int]:
    """Invert simultaneous CDF bounds on an ordered, occupied finite support."""

    lower_index, upper_index = 0, candidate_count - 1
    for index, (lower, upper) in bounds:
        if upper < confidence:
            lower_index = max(lower_index, index + 1)
        if lower >= confidence:
            upper_index = min(upper_index, index)
    if lower_index > upper_index:
        # Includes a sampled contradiction at the maximum support point.
        return 0, candidate_count - 1
    return lower_index, upper_index


def cvar_interval(
    *,
    selected_var: float,
    var_interval: tuple[float, float],
    conditional_interval: tuple[float, float],
    confidence: float,
    support: tuple[float, float],
) -> tuple[float, float]:
    """Bound min_t g(t), where g(t) = t + E[(L-t)+] / (1-alpha).

    g has slopes in [-alpha/(1-alpha), 1]. If the true VaR is in [a,b],
    g(selected) exceeds CVaR by at most max(selected-a,
    alpha/(1-alpha)*(b-selected), 0). The conditional upper bound is already
    an upper bound on the minimum. Intersect with the known support and the
    VaR lower bound. No exact classical/encoded risk value is used here.
    """

    if not all(isfinite(value) for value in conditional_interval):
        return support
    lower_var, upper_var = var_interval
    penalty = max(
        0.0,
        selected_var - lower_var,
        confidence / (1.0 - confidence) * (upper_var - selected_var),
    )
    lower = max(support[0], lower_var, conditional_interval[0] - penalty)
    upper = min(support[1], conditional_interval[1])
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        return support
    return (
        max(support[0], float(np.nextafter(lower, -np.inf))),
        min(support[1], float(np.nextafter(upper, np.inf))),
    )
