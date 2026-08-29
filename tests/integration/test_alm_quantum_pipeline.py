import numpy as np
import pytest

import qfin


def test_alm_scenarios_feed_tested_quantum_cvar_workflow() -> None:
    curve = qfin.YieldCurve([0, 1, 5, 10], [0.02, 0.025, 0.03, 0.035])
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.03)], [10]),
        qfin.LiabilityPortfolio.from_arrays([3, 8], [400, 600]),
        curve,
    )
    scenarios = qfin.RateScenarioSet.parallel(curve, np.linspace(-0.02, 0.02, 16))
    scenario_result = model.run_scenarios(scenarios, engine="numpy")
    problem = qfin.CVaR(scenario_result.loss_distribution(), confidence=0.75)
    compiled = qfin.compile(problem, target_error=0.5, min_qubits=3, max_qubits=4)
    assert isinstance(compiled, qfin.CompiledRiskModel)
    assert compiled.representation.grid_points >= 8
    assert compiled.quantum_algorithm_available
    summary = compiled.run(engine="numpy")
    assert summary.cvar >= summary.var
    runtime = compiled.to_pennylane()
    np.testing.assert_allclose(
        runtime.distribution_probabilities(),
        compiled.representation.probabilities,
        atol=1e-10,
    )
    quantum = compiled.run_quantum(
        shots=3_000,
        schedule=(0, 1, 2, 4),
        seed=17,
        likelihood_grid_size=32_769,
    )
    assert quantum.expected_shortfall == pytest.approx(summary.cvar, abs=0.10)
    assert quantum.resources.total_oracle_queries > 0


def test_problem_capabilities_distinguish_each_implementation_stage() -> None:
    losses = qfin.LossDistribution([0, 1, 2])
    capabilities = qfin.problem_capabilities(qfin.CVaR(losses))
    assert capabilities.financial_model_available
    assert capabilities.quantum_representation_available
    assert capabilities.quantum_algorithm_available
    assert "hybrid" in capabilities.note


def test_system_info_separates_qfin_native_from_lightning() -> None:
    info = qfin.system_info()
    assert info["native_extension"]
    assert info["native_backend"] == "qfin-native"
    assert info["native_cpp_standard"] == "C++20"
    assert info["pennylane_lightning"]
    assert info["preferred_quantum_device"] == "lightning.qubit"
