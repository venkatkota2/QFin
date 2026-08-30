import pytest

import qfin

pytest.importorskip("pennylane")


@pytest.fixture
def small_model() -> qfin.CompiledPricingModel:
    return qfin.compile(
        qfin.EuropeanCall(strike=105, maturity=1.0),
        qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20),
        target_error=1.0,
        min_qubits=3,
        max_qubits=3,
    )


def test_device_target_validates_width_basis_and_connectivity() -> None:
    linear = qfin.DeviceTarget.linear(5)
    assert linear.topology == "linear"
    assert linear.coupling_map == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert not linear.hardware_calibrated

    with pytest.raises(ValueError, match="connect"):
        qfin.DeviceTarget.custom("disconnected", 4, ((0, 1), (2, 3)))
    with pytest.raises(ValueError, match="require RX"):
        qfin.DeviceTarget(
            name="incomplete-basis",
            wires=2,
            coupling_map=((0, 1),),
            topology="custom",
            basis_gates=("RZ", "CNOT"),
        )


def test_gate_decomposition_and_linear_routing_are_explicit(
    small_model: qfin.CompiledPricingModel,
) -> None:
    all_to_all = small_model.device_resources(schedule=(0, 1), shots=100, target="all_to_all")
    linear = small_model.device_resources(schedule=(0, 1), shots=100, target="linear")

    assert all_to_all.total_shots == 200
    assert all(circuit.routing_swaps == 0 for circuit in all_to_all.circuits)
    assert any(circuit.routing_swaps > 0 for circuit in linear.circuits)
    assert linear.maximum_routed_depth > all_to_all.maximum_routed_depth
    assert linear.total_routed_gates_per_objective > (all_to_all.total_routed_gates_per_objective)
    assert linear.total_executed_gates == (
        linear.total_routed_gates_per_objective * linear.shots_per_circuit
    )
    allowed = set(linear.target.coupling_map)
    for circuit in linear.circuits:
        assert set(dict(circuit.gate_types)) <= {"RX", "RY", "RZ", "CNOT"}
        assert set(circuit.routed_two_qubit_edges) <= allowed
        assert circuit.two_qubit_gates == dict(circuit.gate_types).get("CNOT", 0)


def test_structured_loader_can_be_profiled_without_dense_unitaries(
    small_model: qfin.CompiledPricingModel,
) -> None:
    report = small_model.device_resources(
        schedule=(0,),
        shots=10,
        target="linear",
        backend_mode="structured",
    )
    gates = dict(report.circuits[0].gate_types)
    assert "QubitUnitary" not in gates
    assert report.circuits[0].routed_gates > 0


def test_dense_reference_is_rejected_for_hardware_analysis(
    small_model: qfin.CompiledPricingModel,
) -> None:
    with pytest.raises(ValueError, match="numerical reference"):
        small_model.device_resources(backend_mode="dense")


def test_risk_resources_include_hybrid_objective_multiplier() -> None:
    losses = qfin.LossDistribution([0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
    compiled = qfin.compile(
        qfin.VaR(losses, confidence=0.75),
        target_error=0.2,
        min_qubits=2,
        max_qubits=2,
    )
    assert isinstance(compiled, qfin.CompiledRiskModel)
    report = compiled.device_resources(schedule=(0, 1), shots=20, target="linear")
    assert report.objective_evaluations > 1
    assert report.total_circuit_executions == 2 * report.objective_evaluations
    assert report.total_shots == 20 * report.total_circuit_executions


def test_only_tested_runtime_devices_are_accepted(
    small_model: qfin.CompiledPricingModel,
) -> None:
    with pytest.raises(qfin.BackendUnavailableError, match="not registered as tested"):
        small_model.to_pennylane(device_name="lightning.gpu")
    assert "lightning.qubit" in qfin.available_tested_devices()
    info = qfin.system_info()
    assert info["openqasm_export"]
    assert info["noise_simulator"] == "default.mixed"
    assert "default.mixed" in info["tested_quantum_devices"]
