import numpy as np
import pytest

import qfin
import qfin.backends.compressed as compressed_backend_module

pytest.importorskip("pennylane")


def _alm_model() -> qfin.AssetLiabilityModel:
    mortality = qfin.MortalityTable(
        ages=np.arange(40, 101),
        qx=np.concatenate([np.full(60, 0.02), np.array([1.0])]),
    )
    liabilities = qfin.LifePolicyPortfolio(
        (
            qfin.PolicyPosition(
                qfin.TermLifePolicy(
                    issue_age=40,
                    term=30,
                    face_amount=100_000,
                    annual_premium=500,
                ),
                100,
            ),
        )
    )
    assets = qfin.FixedIncomePortfolio(
        (
            qfin.BondPosition(
                qfin.FixedRateBond(1_000, 0.04, 10.0, 2),
                2_000,
            ),
        )
    )
    return qfin.AssetLiabilityModel(
        assets, liabilities, qfin.DiscountCurve.flat(0.04), mortality
    )


def test_uniform_alm_scenarios_use_compressed_lightning_circuit() -> None:
    compiled = qfin.compile_alm(
        _alm_model(),
        [-0.03, -0.01, 0.01, 0.03],
        metric="shortfall_probability",
        target_error=0.10,
    )
    assert compiled.classical_value == pytest.approx(0.5)
    assert compiled.representation.state_preparation_method == (
        "uniform_scenario_hadamard"
    )
    runtime = compiled.to_pennylane()
    assert runtime.device_name == "lightning.qubit"
    assert runtime.probability(0) == pytest.approx(
        compiled.circuit_value / compiled.objective_scale, abs=1e-10
    )
    assert "QubitUnitary" not in runtime.draw(0)


def test_nonuniform_alm_scenarios_use_exact_probability_tree() -> None:
    compiled = qfin.compile_alm(
        _alm_model(),
        [-0.03, 0.0, 0.03],
        probabilities=[0.2, 0.3, 0.5],
        metric="expected_shortfall",
        target_error=20_000,
    )
    assert compiled.payoff_approximation is None
    runtime = compiled.to_pennylane()
    assert isinstance(runtime, qfin.backends.StructuredPennyLaneBackend)
    assert runtime.probability(0) == pytest.approx(
        compiled.classical_value / compiled.objective_scale, abs=1e-10
    )


def test_end_to_end_alm_amplitude_estimation() -> None:
    compiled = qfin.compile_alm(
        _alm_model(),
        [-0.03, -0.01, 0.01, 0.03],
        metric="expected_shortfall",
        target_error=20_000,
    )
    result = compiled.run(shots=4_000, schedule=(0, 1, 2, 4), seed=17)
    assert result.absolute_error < 20_000
    assert result.resources.backend == "pennylane.lightning.qubit"
    assert result.backend == "pennylane.lightning.qubit:compressed"
    assert result.confidence_interval_95[0] <= result.value
    assert result.value <= result.confidence_interval_95[1]
    assert result.to_dict()["metric"] == "expected_shortfall"


def test_schedule_reuses_one_lightning_device(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = qfin.compile_alm(
        _alm_model(),
        [-0.03, -0.01, 0.01, 0.03],
        metric="shortfall_probability",
        target_error=0.10,
    )
    runtime = compiled.to_pennylane()
    original = compressed_backend_module.create_device
    calls = 0

    def counted(*args: object, **kwargs: object) -> tuple[object, str]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(compressed_backend_module, "create_device", counted)
    runtime.run_schedule((0, 1, 2, 4), shots=100, seed=3)
    assert calls == 1
