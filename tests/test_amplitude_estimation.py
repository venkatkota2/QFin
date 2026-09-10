from math import asin, sin, sqrt

import numpy as np
import pytest

from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate
from qfin.algorithms.amplitude_estimation import _dense_reference_maximum_likelihood


def test_mlae_recovers_known_amplitude() -> None:
    amplitude = 0.23
    theta = asin(sqrt(amplitude))
    shots = 100_000
    observations = [
        CircuitObservation(
            power=power,
            successes=round(shots * sin((2 * power + 1) * theta) ** 2),
            shots=shots,
        )
        for power in (0, 1, 2, 4)
    ]
    estimate = maximum_likelihood_amplitude_estimate(observations)
    assert estimate.amplitude == pytest.approx(amplitude, abs=2e-4)
    assert estimate.lower_95 <= estimate.amplitude <= estimate.upper_95
    assert any(
        lower <= estimate.amplitude <= upper
        for lower, upper in estimate.confidence_regions_95
    )
    assert estimate.confidence_method == "likelihood_ratio_with_exact_binomial_guard"
    assert estimate.fisher_interval_95 is not None


@pytest.mark.parametrize("successes, expected", [(0, 0.0), (10, 1.0)])
def test_mlae_likelihood_region_handles_amplitude_boundaries(
    successes: int,
    expected: float,
) -> None:
    estimate = maximum_likelihood_amplitude_estimate(
        [CircuitObservation(power=0, successes=successes, shots=10)]
    )

    assert estimate.amplitude == expected
    assert any(
        lower <= expected <= upper for lower, upper in estimate.confidence_regions_95
    )


def test_mlae_retains_disjoint_regions_for_ambiguous_schedule() -> None:
    estimate = maximum_likelihood_amplitude_estimate(
        [CircuitObservation(power=1, successes=5, shots=10)]
    )

    assert len(estimate.likelihood_ratio_regions_95) == 3
    assert len(estimate.confidence_regions_95) == 3
    assert estimate.lower_95 == estimate.confidence_regions_95[0][0]
    assert estimate.upper_95 == estimate.confidence_regions_95[-1][1]
    assert estimate.to_dict()["confidence_regions_95"] == [
        list(region) for region in estimate.confidence_regions_95
    ]


def test_coarse_to_fine_mlae_agrees_with_dense_reference() -> None:
    generator = np.random.default_rng(20_260_905)
    schedule = (0, 1, 2, 4)
    for _ in range(24):
        amplitude = float(generator.uniform(0.0001, 0.9999))
        theta = asin(sqrt(amplitude))
        shots = int(generator.integers(10, 2_000))
        observations = tuple(
            CircuitObservation(
                power=power,
                successes=int(
                    generator.binomial(shots, sin((2 * power + 1) * theta) ** 2)
                ),
                shots=shots,
            )
            for power in schedule
        )

        estimate = maximum_likelihood_amplitude_estimate(observations)
        dense_theta, dense_log_likelihood = _dense_reference_maximum_likelihood(
            observations
        )

        assert estimate.log_likelihood >= dense_log_likelihood - 1e-9
        assert estimate.amplitude == pytest.approx(sin(dense_theta) ** 2, abs=2e-5)


def test_mlae_validates_schedule() -> None:
    duplicate = [
        CircuitObservation(power=0, successes=4, shots=10),
        CircuitObservation(power=0, successes=5, shots=10),
    ]
    with pytest.raises(ValueError, match="only once"):
        maximum_likelihood_amplitude_estimate(duplicate)


def test_mlae_rejects_search_budget_that_cannot_resolve_schedule() -> None:
    with pytest.raises(ValueError, match="every Grover likelihood interval"):
        maximum_likelihood_amplitude_estimate(
            [CircuitObservation(power=1_000, successes=1, shots=10)], grid_size=1_001,
        )


@pytest.mark.parametrize("schedule", [(0, 2, 5), (0, 1, 16), (1, 3, 7)])
def test_partitioned_search_finds_narrow_modes(schedule: tuple[int, ...]) -> None:
    generator = np.random.default_rng(203)
    for amplitude in (0.000001, 0.01, 0.217, 0.5, 0.991, 0.999999):
        theta = asin(sqrt(amplitude))
        observations = tuple(CircuitObservation(
            power=power, shots=1_000_000,
            successes=int(generator.binomial(1_000_000, sin((2 * power + 1) * theta) ** 2)),
        ) for power in schedule)
        estimate = maximum_likelihood_amplitude_estimate(observations)
        _, reference_likelihood = _dense_reference_maximum_likelihood(observations)
        assert estimate.log_likelihood >= reference_likelihood - 1e-6
        assert estimate.lower_95 <= estimate.amplitude <= estimate.upper_95
