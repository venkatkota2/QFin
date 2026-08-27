import numpy as np
import pytest

import qfin


def test_fixed_lognormal_encoding_is_a_valid_quantum_state() -> None:
    distribution = qfin.LogNormal(mu=np.log(100), sigma=0.2)
    encoding = qfin.encode(distribution, qubits=5, min_qubits=3, max_qubits=8)
    assert encoding.qubits == 5
    assert encoding.grid_points == 32
    assert np.sum(encoding.probabilities) == pytest.approx(1.0)
    assert np.linalg.norm(encoding.state_vector()) == pytest.approx(1.0)
    assert encoding.lower_bound > 0
    assert 0 <= encoding.tail_probability < 1e-4


def test_automatic_encoding_uses_financial_objective_units() -> None:
    distribution = qfin.Normal(mean_value=0, standard_deviation=1)
    encoding = qfin.encode(
        distribution,
        target_error=0.02,
        objective=lambda grid: np.maximum(grid, 0),
        min_qubits=2,
        max_qubits=8,
    )
    assert 2 <= encoding.qubits <= 8
    assert encoding.objective == "callable"
    assert encoding.discretization_error <= 0.02


def test_empirical_encoding_preserves_the_mean() -> None:
    distribution = qfin.EmpiricalDistribution(np.array([1.0, 2.0, 10.0]))
    encoding = qfin.encode(distribution, qubits=3, min_qubits=2, max_qubits=8)
    assert encoding.conditional_mean == pytest.approx(distribution.mean)


def test_empirical_encoding_does_not_fold_omitted_mass_into_boundary_bins() -> None:
    distribution = qfin.EmpiricalDistribution(
        np.array([0.0, 1.0, 2.0, 100.0]),
        probabilities=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    encoding = qfin.encode(
        distribution,
        qubits=2,
        min_qubits=2,
        max_qubits=2,
        bounds=(0.0, 2.0),
    )

    assert encoding.tail_probability == pytest.approx(0.4)
    assert encoding.conditional_mean == pytest.approx((0.2 + 0.6) / 0.6)


def test_quantile_encoding_has_parameter_free_uniform_probabilities() -> None:
    distribution = qfin.LogNormal(mu=np.log(100), sigma=0.2)
    encoding = qfin.encode_quantiles(
        distribution,
        qubits=5,
        min_qubits=3,
        max_qubits=8,
    )
    np.testing.assert_allclose(encoding.probabilities, np.full(32, 1 / 32))
    assert encoding.encoding_method == "inverse_cdf_quantile"
    assert encoding.state_preparation_method == "uniform_quantile_hadamard"
    assert np.all(np.diff(encoding.grid) > 0)
