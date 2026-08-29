from math import asin, sin, sqrt

import numpy as np
import pytest

import qfin

pytest.importorskip("pennylane")


@pytest.fixture
def losses() -> qfin.LossDistribution:
    return qfin.LossDistribution(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]))


def _compile(problem: object) -> qfin.CompiledRiskModel:
    model = qfin.compile(problem, target_error=0.5, min_qubits=3, max_qubits=3)
    assert isinstance(model, qfin.CompiledRiskModel)
    return model


def test_tail_objective_state_and_grover_probability(
    losses: qfin.LossDistribution,
) -> None:
    model = _compile(qfin.TailProbability(losses, threshold=5.0))
    objective = qfin.tail_probability_objective(model.representation, 5.0)
    runtime = model.to_pennylane()

    np.testing.assert_allclose(
        runtime.distribution_probabilities(),
        model.representation.probabilities,
        atol=1e-10,
    )
    assert runtime.theoretical_amplitude(objective) == pytest.approx(0.25)
    assert runtime.probability(objective, power=0) == pytest.approx(0.25, abs=1e-10)
    theta = asin(sqrt(0.25))
    assert runtime.probability(objective, power=1) == pytest.approx(sin(3 * theta) ** 2, abs=1e-10)
    assert "QubitUnitary" not in runtime.draw(objective)


def test_tail_probability_quantum_workflow(losses: qfin.LossDistribution) -> None:
    model = _compile(qfin.TailProbability(losses, threshold=5.0))
    result = model.run_quantum(
        shots=2_000,
        schedule=(0, 1, 2),
        seed=5,
        likelihood_grid_size=32_769,
    )

    assert result.value == pytest.approx(0.25, abs=0.01)
    assert result.tail_probability == result.value
    assert result.confidence_interval_95[0] <= result.value
    assert result.value <= result.confidence_interval_95[1]
    assert result.resources.objective_evaluations == 1
    assert result.resources.total_shots == 6_000
    assert result.backend == "pennylane.lightning.qubit:structured"


def test_quantum_var_binary_search_and_resources(losses: qfin.LossDistribution) -> None:
    model = _compile(qfin.VaR(losses, confidence=0.70))
    result = model.run_quantum(
        shots=2_000,
        schedule=(0, 1, 2),
        seed=9,
        likelihood_grid_size=32_769,
        bootstrap_resamples=40,
        bootstrap_seed=3,
    )

    assert result.value == pytest.approx(5.0)
    assert result.value_at_risk == result.value
    assert result.classical_value == pytest.approx(5.0)
    assert len(result.amplitude_estimates) >= 2
    assert result.resources.threshold_evaluations == len(result.amplitude_estimates)
    assert result.classical_interval is not None
    assert "binary search" in model.explain()


def test_quantum_cvar_tail_excess_is_close_to_discrete_reference(
    losses: qfin.LossDistribution,
) -> None:
    model = _compile(qfin.CVaR(losses, confidence=0.70))
    result = model.run_quantum(
        shots=4_000,
        schedule=(0, 1, 2, 4),
        seed=11,
        likelihood_grid_size=32_769,
    )

    assert result.value_at_risk == pytest.approx(5.0)
    assert result.expected_shortfall == pytest.approx(13.3333333333, abs=0.15)
    assert result.classical_value == pytest.approx(13.3333333333)
    assert result.resources.excess_evaluations == 1
    assert result.resources.objective_evaluations == len(result.amplitude_estimates)
    assert "O(2**data_qubits)" in result.resources.state_preparation_complexity
    assert result.to_dict()["caveat"]


def test_risk_compiler_capabilities_and_classical_backend(
    losses: qfin.LossDistribution,
) -> None:
    problem = qfin.CVaR(losses, confidence=0.70)
    capabilities = qfin.problem_capabilities(problem)
    assert capabilities.quantum_representation_available
    assert capabilities.quantum_algorithm_available
    assert "hybrid" in capabilities.note

    model = qfin.compile(
        problem,
        backend="classical",
        target_error=0.5,
        min_qubits=3,
        max_qubits=3,
    )
    assert isinstance(model, qfin.CompiledRiskModel)
    assert model.run(engine="numpy").cvar == pytest.approx(13.3333333333)
    with pytest.raises(ValueError, match="not 'pennylane'"):
        model.to_pennylane()


def test_risk_device_override_and_logical_resource_caveat(
    losses: qfin.LossDistribution,
) -> None:
    model = _compile(qfin.VaR(losses, confidence=0.70))
    runtime = model.to_pennylane(device_name="default.qubit")
    assert runtime.device_name == "default.qubit"
    resources = model.resources(schedule=(0, 1), shots=100, device_name="default.qubit")
    assert resources.per_objective.backend == "pennylane.default.qubit"
    assert resources.estimated_sort_comparisons > 0
    assert "not a hardware runtime" in str(resources.to_dict()["caveat"])


def test_risk_objective_matches_default_qubit_and_lightning(
    losses: qfin.LossDistribution,
) -> None:
    model = _compile(qfin.TailProbability(losses, threshold=5.0))
    objective = qfin.tail_probability_objective(model.representation, 5.0)
    default_probability = model.to_pennylane(
        device_name="default.qubit"
    ).probability(objective, power=2)
    lightning_probability = model.to_pennylane(
        device_name="lightning.qubit"
    ).probability(objective, power=2)
    assert lightning_probability == pytest.approx(default_probability, abs=1e-10)
