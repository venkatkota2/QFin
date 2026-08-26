import numpy as np
import pytest

import qfin
from qfin.circuits import (
    PayoffRotation,
    ProbabilityTreePreparation,
    UniformQuantilePreparation,
    WalshPayoffApproximation,
    probability_tree_angles,
)

pytest.importorskip("pennylane")


def test_probability_tree_has_one_angle_per_internal_node() -> None:
    probabilities = np.array([0.05, 0.15, 0.30, 0.10, 0.12, 0.08, 0.07, 0.13])
    levels = probability_tree_angles(probabilities)
    assert [level.size for level in levels] == [1, 2, 4]
    tree = ProbabilityTreePreparation.from_probabilities(probabilities)
    assert tree.qubits == 3
    assert tree.rotation_count == 7


def test_probability_tree_reconstructs_distribution() -> None:
    probabilities = np.array([0.05, 0.15, 0.30, 0.10, 0.12, 0.08, 0.07, 0.13])
    encoding = qfin.DistributionEncoding(
        grid=np.arange(8, dtype=float),
        probabilities=probabilities,
        qubits=3,
        lower_bound=0,
        upper_bound=8,
        tail_probability=0,
        discretization_error=0,
        mean_error=0,
        objective="expectation",
    )
    payoff = np.linspace(0, 1, 8)
    from qfin.backends import StructuredPennyLaneBackend

    backend = StructuredPennyLaneBackend(encoding, payoff)
    np.testing.assert_allclose(
        backend.distribution_probabilities(), probabilities, atol=1e-12
    )


def test_payoff_angles_encode_success_probabilities() -> None:
    payoff = np.array([0.0, 0.25, 0.5, 1.0])
    loader = PayoffRotation.from_normalized_payoff(payoff)
    np.testing.assert_allclose(np.sin(loader.angles / 2) ** 2, payoff, atol=1e-12)
    assert loader.rotation_count == 4


def test_uniform_quantile_loader_has_no_parameters() -> None:
    loader = UniformQuantilePreparation(qubits=5)
    assert loader.gate_count == 5
    assert loader.parameter_count == 0


def test_full_walsh_payoff_reconstructs_rotation_angles() -> None:
    payoff = np.array([0.0, 0.05, 0.2, 0.6, 0.1, 0.4, 0.8, 1.0])
    approximation = WalshPayoffApproximation.fit(
        payoff,
        financial_multiplier=1.0,
        target_price_error=1e-12,
        max_angle_rmse=1e-12,
    )
    target = 2 * np.arcsin(np.sqrt(payoff))
    np.testing.assert_allclose(approximation.approximate_angles(), target, atol=1e-12)
    assert approximation.met_tolerance
