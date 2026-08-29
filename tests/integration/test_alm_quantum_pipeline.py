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


def test_multiperiod_alm_and_life_scenarios_preserve_quantum_risk_bridge() -> None:
    curve = qfin.YieldCurve([0, 1, 5], [0.02, 0.025, 0.03])
    scenarios = qfin.EconomicScenarioSet(
        np.linspace(-0.01, 0.01, 8)[:, None, None] * np.ones((8, 3, curve.times.size)),
        equity_returns=np.linspace(-0.05, 0.05, 8)[:, None] * np.ones((8, 3)),
        inflation_rates=0.02,
    )
    alm = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.03)], [10], equity_value=200),
        qfin.LiabilityPortfolio.from_arrays([2, 4], [400, 600], inflation_linkage=[1, 1]),
        curve,
    )
    alm_paths = alm.project_paths(scenarios, engine="native", scenario_chunk_size=3)
    alm_problem = qfin.CVaR(alm_paths.loss_distribution(), confidence=0.75)
    alm_compiled = qfin.compile(
        alm_problem, backend="classical", target_error=0.5, min_qubits=3, max_qubits=3
    )
    assert alm_compiled.run().cvar >= alm_compiled.run().var

    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [0.01, 0.01]),
        curve,
        lapse_rate=0.03,
    )
    life = qfin.project_liability_scenarios(
        [qfin.LifePolicy(40, 100_000, 500, 3)],
        assumptions,
        scenarios,
        engine="native",
        scenario_chunk_size=3,
    )
    life_problem = qfin.VaR(life.loss_distribution(), confidence=0.75)
    life_compiled = qfin.compile(
        life_problem, backend="classical", target_error=0.5, min_qubits=3, max_qubits=3
    )
    assert np.isfinite(life_compiled.run().var)
    assert qfin.problem_capabilities(life).quantum_representation_available
