import numpy as np
import pytest

import qfin


@pytest.fixture
def compiled_call() -> qfin.CompiledPricingModel:
    market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
    option = qfin.EuropeanCall(strike=105, maturity=1.0)
    return qfin.compile(option, market, target_error=0.10, max_qubits=8)


def test_compiler_builds_complete_pricing_problem(
    compiled_call: qfin.CompiledPricingModel,
) -> None:
    model = compiled_call
    assert 3 <= model.representation.qubits <= 8
    assert np.all((model.normalized_payoff >= 0) & (model.normalized_payoff <= 1))
    assert model.algorithm_name == "maximum_likelihood_amplitude_estimation"
    assert model.representation.encoding_method == "inverse_cdf_quantile"
    assert model.payoff_approximation is not None
    assert model.payoff_approximation.met_tolerance
    assert model.compilation_converged
    assert model.payoff_approximation.parameter_count < model.representation.grid_points
    assert model.classical_value > 0
    assert model.representation_error < 0.25
    assert "Black-Scholes" in model.explain()


def test_resource_report_is_explicit_about_mlae(
    compiled_call: qfin.CompiledPricingModel,
) -> None:
    resources = compiled_call.resources(schedule=(0, 1, 2), shots=100)
    assert resources.total_logical_qubits == compiled_call.representation.qubits + 2
    assert resources.work_qubits == 1
    assert resources.estimation_qubits == 0
    assert resources.total_shots == 300
    assert resources.oracle_queries == 900
    assert resources.backend_mode == "compressed"
    assert resources.distribution_rotations == 0
    assert resources.distribution_gates == compiled_call.representation.qubits
    assert resources.classical_parameter_count == (
        compiled_call.payoff_approximation.parameter_count
    )
    assert resources.dense_unitary_entries == 0
    assert "before device-specific" in str(resources.to_dict()["caveat"])


def test_unsupported_backend_is_rejected() -> None:
    market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
    option = qfin.EuropeanCall(strike=105, maturity=1.0)
    with pytest.raises(qfin.CompilationError, match="pennylane"):
        qfin.compile(option, market, backend="qiskit")


def test_compiler_reports_when_payoff_term_cap_misses_tolerance() -> None:
    market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
    option = qfin.EuropeanCall(strike=105, maturity=1.0)
    model = qfin.compile(
        option,
        market,
        target_error=0.10,
        max_qubits=8,
        payoff_max_terms=4,
    )
    assert model.payoff_approximation is not None
    assert not model.payoff_approximation.met_tolerance
    assert not model.compilation_converged
    assert "not met" in model.explain()


def test_compiler_does_not_false_converge_when_coarse_grids_miss_tail_payoff() -> None:
    market = qfin.BlackScholes(spot=100, rate=0.03, volatility=0.50)
    option = qfin.EuropeanCall(strike=300, maturity=1.0)
    model = qfin.compile(option, market, target_error=0.10, max_qubits=12)
    representation_budget = (
        model.error_budget.domain_truncation + model.error_budget.discretization
    )

    assert model.classical_value > 0.40
    assert model.representation.qubits > 4
    assert model.payoff_scale > 0
    assert model.representation_error <= representation_budget
    assert model.representation_converged
    assert "Representation validation error" in model.explain()
