"""Analytical bounds and exhaustive finite-count coverage, not fitted references."""

from dataclasses import replace
from itertools import product
from math import asin, sin, sqrt

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate
from qfin.algorithms.amplitude_estimation import _exact_binomial_confidence_regions
from qfin.compiler._risk_uncertainty import (
    cvar_interval,
    objective_budget,
    quantile_indices,
    simultaneous_amplitude_interval,
)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 8, 9, 1000])
def test_budget_bounds_every_binary_search_path(count):
    def visit(lo, hi, visited):
        if lo == hi:
            return visited + int(lo not in indices)
        mid = (lo + hi) // 2
        indices.add(mid)
        left = visit(lo, mid, visited + 1)
        right = visit(mid + 1, hi, visited + 1)
        indices.remove(mid)
        return max(left, right)

    indices = set()
    assert visit(0, count - 1, 0) <= objective_budget(count)
    assert objective_budget(count, 5) == objective_budget(count) + 5


@pytest.mark.parametrize("successes", range(9))
def test_allocated_binomial_bound_matches_independent_scipy_exact_test(successes):
    estimate = maximum_likelihood_amplitude_estimate([CircuitObservation(0, successes, 8)])
    lower, upper = simultaneous_amplitude_interval(estimate, 7)
    reference = binomtest(successes, 8).proportion_ci(confidence_level=1 - 0.05 / 7)
    assert lower == pytest.approx(reference.low, abs=1e-12)
    assert upper == pytest.approx(reference.high, abs=1e-12)


@pytest.mark.parametrize("schedule", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("amplitude", [0.0, 1e-6, 0.05, 0.25, 0.5, 0.95, 1.0])
def test_exact_guard_exhaustive_binomial_coverage(schedule, amplitude):
    # Enumerate all count combinations, weighting by exact binomial probabilities.
    shots, budget = 5, 4
    theta = asin(sqrt(amplitude))
    covered = 0.0
    probabilities = [sin((2 * k + 1) * theta) ** 2 for k in schedule]
    for successes in product(range(shots + 1), repeat=len(schedule)):
        probability = float(
            np.prod([binom.pmf(k, shots, p) for k, p in zip(successes, probabilities, strict=True)])
        )
        observations = [
            CircuitObservation(power, k, shots)
            for power, k in zip(schedule, successes, strict=True)
        ]
        regions = _exact_binomial_confidence_regions(
            observations, failure_probability=0.05 / budget
        )
        if not regions or any(lo - 1e-14 <= theta <= hi + 1e-14 for lo, hi in regions):
            covered += probability
    assert covered >= 1 - 0.05 / budget - 1e-12


def test_empty_exact_intersection_falls_back_to_full_amplitude_support():
    estimate = maximum_likelihood_amplitude_estimate([CircuitObservation(0, 0, 10000)])
    inconsistent = replace(
        estimate,
        observations=(
            CircuitObservation(0, 0, 10000),
            CircuitObservation(1, 10000, 10000),
        ),
    )
    assert simultaneous_amplitude_interval(inconsistent, 2) == (0, 1)


@pytest.mark.parametrize("failure_probability", [0, 1, float("nan")])
def test_invalid_exact_guard_budget_is_rejected(failure_probability):
    with pytest.raises(ValueError, match="failure probability"):
        _exact_binomial_confidence_regions(
            [CircuitObservation(0, 0, 1)], failure_probability=failure_probability
        )


def test_contradictory_cdf_bounds_do_not_collapse_to_a_selected_point():
    assert quantile_indices(4, 0.9, [(0, (0.95, 1)), (2, (0, 0.1))]) == (0, 3)
    assert quantile_indices(4, 0.9, [(3, (0, 0.1))]) == (0, 3)
    assert quantile_indices(4, 0.9, [(0, (0, 0.8)), (2, (0.95, 1))]) == (1, 2)


@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95, 0.999])
def test_cvar_propagation_contains_exact_discrete_functional(confidence):
    losses = np.array([-30, -2, 0, 5, 1000.0])
    weights = np.array([0.01, 0.2, 0.6, 0.189, 0.001])
    cumulative = np.cumsum(weights)
    true_var = losses[np.searchsorted(cumulative, confidence)]
    # Independent quantile integral with fractional mass at alpha.
    previous = np.r_[0, cumulative[:-1]]
    mass = np.maximum(0, cumulative - np.maximum(previous, confidence))
    exact_es = float(np.dot(losses, mass) / (1 - confidence))
    for selected, a, b in product(losses, losses, losses):
        if a > true_var or b < true_var:
            continue
        g = float(selected + np.dot(weights, np.maximum(losses - selected, 0)) / (1 - confidence))
        lo, hi = cvar_interval(
            selected_var=float(selected),
            var_interval=(float(a), float(b)),
            conditional_interval=(g - 0.1, g + 0.1),
            confidence=confidence,
            support=(-30, 1000),
        )
        assert lo - 1e-9 <= exact_es <= hi + 1e-9


def test_cvar_collapsed_wrong_maximum_threshold_gets_full_support():
    assert cvar_interval(
        selected_var=1000,
        var_interval=(0, 1000),
        conditional_interval=(1000, 1000),
        confidence=0.95,
        support=(0, 1000),
    ) == (0, 1000)


@pytest.mark.parametrize("conditional", [(3000, 3100), (float("nan"), float("nan"))])
def test_inconsistent_cvar_bounds_fall_back_to_support(conditional):
    assert cvar_interval(
        selected_var=1000,
        var_interval=(0, 1000),
        conditional_interval=conditional,
        confidence=0.95,
        support=(0, 1000),
    ) == (0, 1000)
