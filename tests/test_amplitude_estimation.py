from math import asin, sin, sqrt

import pytest

from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate


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


def test_mlae_validates_schedule() -> None:
    duplicate = [
        CircuitObservation(power=0, successes=4, shots=10),
        CircuitObservation(power=0, successes=5, shots=10),
    ]
    with pytest.raises(ValueError, match="only once"):
        maximum_likelihood_amplitude_estimate(duplicate)

