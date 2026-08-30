import numpy as np
import pytest

import qfin


def test_factorized_encoding_avoids_joint_probability_table() -> None:
    encoding = qfin.encode_independent_factors(
        [qfin.Normal(), qfin.Normal(), qfin.Normal()],
        qubits_per_factor=4,
        factor_names=["rate", "equity", "inflation"],
    )

    assert encoding.total_qubits == 12
    assert encoding.joint_grid_points == 4_096
    assert encoding.stored_marginal_points == 48
    assert encoding.avoids_joint_probability_table
    with pytest.raises(ValueError, match="above max_points"):
        encoding.materialize(max_points=1_000)

    report = qfin.compare_state_preparation_strategies(encoding)
    selected = report.require_selected()
    flattened = next(
        candidate
        for candidate in report.candidates
        if candidate.strategy == "flattened_probability_tree"
    )
    assert selected.strategy == "factorized_marginal_loader"
    assert selected.classical_parameters == 0
    assert flattened.classical_parameters == 4_095
    assert flattened.requires_joint_materialization


def test_small_factor_grid_materialization_is_a_validation_oracle() -> None:
    encoding = qfin.encode_independent_factors(
        [qfin.Normal(), qfin.LogNormal(mu=0.0, sigma=0.2)],
        qubits_per_factor=(2, 3),
        factor_names=("normal", "lognormal"),
    )
    materialized = encoding.materialize(max_points=32)

    assert materialized.values.shape == (32, 2)
    assert materialized.probabilities.shape == (32,)
    assert np.sum(materialized.probabilities) == pytest.approx(1.0)
    expected = encoding.expectation(lambda values: values[:, 0] + values[:, 1])
    direct = float(np.dot(materialized.probabilities, np.sum(materialized.values, axis=1)))
    assert expected == pytest.approx(direct)


def test_factorized_loader_matches_small_materialized_distribution() -> None:
    qml = pytest.importorskip("pennylane")
    encoding = qfin.encode_independent_factors(
        [qfin.Normal(), qfin.Normal()],
        qubits_per_factor=2,
    )
    loader = qfin.FactorizedPreparation.from_encoding(encoding)
    device = qml.device("default.qubit", wires=loader.total_qubits)

    @qml.qnode(device)  # type: ignore[untyped-decorator]
    def circuit() -> object:
        loader.apply(range(loader.total_qubits))
        return qml.probs(wires=range(loader.total_qubits))

    probabilities = np.asarray(circuit(), dtype=np.float64)
    expected = encoding.materialize(max_points=16).probabilities
    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    assert loader.parameter_count == 0
    assert loader.gate_count == 4
    assert not loader.to_dict()["joint_angle_table_constructed"]


def test_probability_marginals_use_sum_not_product_of_angle_tables() -> None:
    encoding = qfin.encode_independent_factors(
        [qfin.LogNormal(mu=0.0, sigma=0.2), qfin.LogNormal(mu=0.1, sigma=0.3)],
        qubits_per_factor=3,
        method="probability",
    )
    loader = qfin.FactorizedPreparation.from_encoding(encoding)
    assert loader.parameter_count == 14
    assert loader.parameter_count < encoding.joint_grid_points - 1


def test_gaussian_factor_encoding_preserves_correlation_in_small_oracle() -> None:
    model = qfin.GaussianFactorModel(
        ("rates", "equity"),
        np.array([[1.0, 0.35], [0.35, 1.0]]),
        means=np.array([0.01, 0.04]),
        standard_deviations=np.array([0.02, 0.15]),
    )
    encoding = qfin.encode_gaussian_factors(model, qubits_per_factor=4)
    materialized = encoding.materialize(max_points=256)
    weighted_mean = materialized.probabilities @ materialized.values
    centered = materialized.values - weighted_mean
    covariance = (centered * materialized.probabilities[:, None]).T @ centered
    correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))

    assert materialized.value_names == ("rates", "equity")
    np.testing.assert_allclose(weighted_mean, [0.01, 0.04], atol=1e-12)
    assert correlation[0, 1] == pytest.approx(0.35, abs=1e-12)
    assert encoding.transform is not None
    assert not encoding.transform.to_dict()["quantum_arithmetic_implemented"]


def test_target_feedback_can_reject_or_cap_a_representation() -> None:
    encoding = qfin.encode_independent_factors(
        [qfin.Normal(), qfin.Normal()],
        qubits_per_factor=3,
    )
    too_small = qfin.compare_state_preparation_strategies(
        encoding,
        target=qfin.DeviceTarget.linear(5),
    )
    assert too_small.selected is None
    with pytest.raises(qfin.ResourceLimitError, match="no implemented portable"):
        too_small.require_selected()

    option = qfin.EuropeanCall(strike=105, maturity=1.0)
    market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
    compiled = qfin.compile(
        option,
        market,
        target_error=1.0,
        min_qubits=3,
        max_qubits=8,
        representation_target=qfin.DeviceTarget.linear(6),
    )
    assert isinstance(compiled, qfin.CompiledPricingModel)
    assert compiled.representation.qubits <= 4
    assert compiled.state_preparation_strategy.target_name == "research-linear-6q"


def test_auto_risk_backend_falls_back_when_target_cannot_fit_loader() -> None:
    problem = qfin.VaR(qfin.LossDistribution(np.array([0.0, 1.0, 2.0, 3.0])), confidence=0.75)
    target = qfin.DeviceTarget.linear(4)
    compiled = qfin.compile(
        problem,
        backend="auto",
        min_qubits=3,
        max_qubits=3,
        representation_target=target,
    )
    assert isinstance(compiled, qfin.CompiledRiskModel)
    assert compiled.backend_name == "classical"
    assert compiled.state_preparation_strategy.selected is None
    with pytest.raises(ValueError, match="not 'pennylane'"):
        compiled.to_pennylane()

    with pytest.raises(qfin.ResourceLimitError):
        qfin.compile(
            problem,
            backend="pennylane",
            min_qubits=3,
            max_qubits=3,
            representation_target=target,
        )


def test_system_info_reports_milestone_capability_boundaries() -> None:
    info = qfin.system_info()

    assert info["qfin_version"] == "0.8.0"
    assert info["factorized_state_preparation"] is True
    assert info["portfolio_optimization"] == "classical-scipy"
    assert info["block_encoding_implemented"] is False
    assert info["qsvt_implemented"] is False
