import numpy as np
import pytest

import qfin


def test_alm_scenarios_feed_loss_representation_without_fake_quantum_claim() -> None:
    curve = qfin.YieldCurve([0, 1, 5, 10], [0.02, 0.025, 0.03, 0.035])
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(5, 0.03)], [10]),
        qfin.LiabilityPortfolio.from_arrays([3, 8], [400, 600]),
        curve,
    )
    scenarios = qfin.RateScenarioSet.parallel(curve, np.linspace(-0.02, 0.02, 16))
    scenario_result = model.run_scenarios(scenarios, engine="numpy")
    problem = qfin.CVaR(scenario_result.loss_distribution(), confidence=0.95)
    compiled = qfin.compile(problem, target_error=1.0, min_qubits=2, max_qubits=6)
    assert isinstance(compiled, qfin.CompiledRiskModel)
    assert compiled.representation.grid_points >= 4
    assert not compiled.quantum_algorithm_available
    summary = compiled.run(engine="numpy")
    assert summary.cvar >= summary.var
    with pytest.raises(qfin.CompilationError, match="not implemented"):
        compiled.to_pennylane()


def test_problem_capabilities_distinguish_each_implementation_stage() -> None:
    losses = qfin.LossDistribution([0, 1, 2])
    capabilities = qfin.problem_capabilities(qfin.CVaR(losses))
    assert capabilities.financial_model_available
    assert capabilities.quantum_representation_available
    assert not capabilities.quantum_algorithm_available
    assert "not yet implemented" in capabilities.note


def test_system_info_separates_qfin_native_from_lightning() -> None:
    info = qfin.system_info()
    assert info["native_extension"]
    assert info["native_backend"] == "qfin-native"
    assert info["native_cpp_standard"] == "C++20"
    assert info["pennylane_lightning"]
    assert info["preferred_quantum_device"] == "lightning.qubit"
