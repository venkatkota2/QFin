from math import asin, sin, sqrt

import pytest

import qfin

pytest.importorskip("pennylane")


@pytest.fixture
def small_model() -> qfin.CompiledPricingModel:
    market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
    option = qfin.EuropeanCall(strike=105, maturity=1.0)
    return qfin.compile(
        option,
        market,
        target_error=1.0,
        min_qubits=3,
        max_qubits=3,
    )


def test_pennylane_circuit_encodes_expected_amplitude(
    small_model: qfin.CompiledPricingModel,
) -> None:
    backend = small_model.to_pennylane()
    amplitude = backend.theoretical_amplitude()
    assert backend.probability(power=0) == pytest.approx(amplitude, abs=1e-10)
    theta = asin(sqrt(amplitude))
    assert backend.probability(power=1) == pytest.approx(sin(3 * theta) ** 2, abs=1e-10)


def test_structured_and_dense_backends_are_numerically_equivalent(
    small_model: qfin.CompiledPricingModel,
) -> None:
    structured = small_model.to_pennylane(mode="structured")
    dense = small_model.to_pennylane(mode="dense")
    for power in (0, 1, 2):
        assert structured.probability(power) == pytest.approx(dense.probability(power), abs=1e-10)


def test_default_circuit_contains_no_dense_unitaries(
    small_model: qfin.CompiledPricingModel,
) -> None:
    backend = small_model.to_pennylane()
    assert backend.device_name == "lightning.qubit"
    diagram = backend.draw(power=1)
    assert "RZ" in diagram or "PauliRot" in diagram
    assert " H " in diagram or "─H─" in diagram
    assert "MultiControlledX" not in diagram  # Drawn as a controlled X symbol.
    assert "QubitUnitary" not in diagram
    specs = backend.circuit_specs(power=1)
    assert "QubitUnitary" not in specs["gate_types"]
    assert specs["num_gates"] > 0


def test_compressed_backend_matches_its_reported_amplitude(
    small_model: qfin.CompiledPricingModel,
) -> None:
    backend = small_model.to_pennylane(mode="compressed")
    assert backend.probability(0) == pytest.approx(backend.theoretical_amplitude(), abs=1e-10)
    assert backend.distribution_loader.parameter_count == 0


def test_end_to_end_quantum_run_is_close_to_discrete_value(
    small_model: qfin.CompiledPricingModel,
) -> None:
    result = small_model.run(shots=4_000, schedule=(0, 1, 2, 4), seed=11)
    assert result.target_error_unit == "currency / price units"
    assert result.to_dict()["target_error_unit"] == "currency / price units"
    assert result.estimation_error < 1.0
    assert result.confidence_interval_95[0] <= result.value
    assert result.value <= result.confidence_interval_95[1]
    assert result.resources.total_shots == 16_000
    assert result.resources.backend_mode == "compressed"
    assert result.resources.backend == "pennylane.lightning.qubit"
    assert result.backend == "pennylane.lightning.qubit:compressed"
    assert result.payoff_approximation is not None
    assert result.to_dict()["payoff_approximation"] is not None
    assert result.estimation_error == pytest.approx(abs(result.value - result.circuit_value))


def test_pennylane_device_can_be_overridden(
    small_model: qfin.CompiledPricingModel,
) -> None:
    backend = small_model.to_pennylane(device_name="default.qubit")
    assert backend.device_name == "default.qubit"
    resources = small_model.resources(device_name="default.qubit")
    assert resources.backend == "pennylane.default.qubit"


def test_auto_device_falls_back_when_lightning_is_absent(
    small_model: qfin.CompiledPricingModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qfin.backends.devices as devices

    monkeypatch.setattr(
        devices,
        "find_spec",
        lambda name: None if name == "pennylane_lightning" else object(),
    )
    backend = small_model.to_pennylane()
    assert backend.device_name == "default.qubit"
    assert small_model.resources().backend == "pennylane.default.qubit"


@pytest.mark.parametrize("seed", [19, 127, 811])
def test_seeded_backend_matrix_preserves_the_encoded_objective(seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    market = qfin.BlackScholes(100, float(rng.uniform(-.03, .10)),
                              float(rng.uniform(.1, .5)))
    option = qfin.EuropeanPut(float(rng.uniform(80, 120)), 1)
    model = qfin.compile(option, market, target_error=.01, min_qubits=3, max_qubits=3,
                         payoff_angle_tolerance=1e-12)
    dense = model.to_pennylane(mode="dense", device_name="default.qubit")
    for device in ("default.qubit", "lightning.qubit"):
        if device == "lightning.qubit" and not qfin.system_info()["pennylane_lightning"]:
            continue
        for mode in ("dense", "structured", "compressed"):
            runtime = model.to_pennylane(mode=mode, device_name=device)
            for power in (0, 1, 3):
                assert runtime.probability(power) == pytest.approx(
                    dense.probability(power), abs=2e-12,
                )
            first = runtime.run_schedule(schedule=(0, 1), shots=20, seed=seed)
            second = runtime.run_schedule(schedule=(0, 1), shots=20, seed=seed)
            assert first == second
