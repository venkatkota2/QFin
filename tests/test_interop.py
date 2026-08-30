from hashlib import sha256

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


def test_openqasm_export_is_routed_reproducible_and_self_describing(
    small_model: qfin.CompiledPricingModel,
) -> None:
    exported = small_model.to_openqasm(power=1, target="linear")
    repeated = small_model.to_openqasm(power=1, target="linear")

    assert exported.program.startswith("OPENQASM 2.0;")
    assert 'include "qelib1.inc";' in exported.program
    assert "qreg q[5];" in exported.program
    assert exported.sha256 == sha256(exported.program.encode()).hexdigest()
    assert exported.sha256 == repeated.sha256
    assert exported.resources.routing_swaps > 0
    assert exported.to_dict()["format"] == "OpenQASM 2.0"
    assert "program" not in exported.to_dict()
    assert exported.to_dict(include_program=True)["program"] == exported.program


def test_qiskit_export_round_trips_the_targeted_gate_set(
    small_model: qfin.CompiledPricingModel,
) -> None:
    qiskit = pytest.importorskip("qiskit")
    circuit = small_model.to_qiskit(power=1, target="linear")
    operations = {item.operation.name for item in circuit.data}
    assert qiskit.__version__.startswith("2.")
    assert circuit.num_qubits == 5
    assert operations <= {"rx", "ry", "rz", "cx", "measure"}
    assert (
        circuit.size()
        == small_model.to_openqasm(power=1, target="linear").resources.routed_gates + 5
    )


def test_routed_export_preserves_the_objective_probability(
    small_model: qfin.CompiledPricingModel,
) -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    exported = small_model.to_openqasm(power=1, target="linear")
    circuit = small_model.to_qiskit(power=1, target="linear")
    state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    probabilities = state.probabilities([exported.resources.objective_physical_wire])
    assert probabilities[1] == pytest.approx(small_model.to_pennylane().probability(1), abs=1e-10)


def test_missing_qiskit_dependency_has_an_actionable_error(
    small_model: qfin.CompiledPricingModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qfin.backends.interop as interop

    monkeypatch.setattr(interop, "find_spec", lambda _name: None)
    with pytest.raises(qfin.BackendUnavailableError, match="optional dependency"):
        small_model.to_qiskit()


class _CouplingMap:
    def get_edges(self) -> list[tuple[int, int]]:
        return [(0, 1), (1, 2), (2, 3), (3, 4)]


class _Backend:
    name = "test-backend"
    num_qubits = 5
    operation_names = ("rz", "sx", "x", "cx", "reset", "measure", "if_else")
    coupling_map = _CouplingMap()


def test_provider_capability_inspection_does_not_submit_work() -> None:
    report = qfin.inspect_qiskit_backend(_Backend(), required_wires=5)
    assert report.backend_name == "test-backend"
    assert report.coupling_connected
    assert report.has_entangling_gate
    assert report.has_measurement
    assert report.supports_dynamic_circuits
    assert report.qfin_export_compatible
    assert not qfin.inspect_qiskit_backend(_Backend(), required_wires=6).qfin_export_compatible


def test_risk_model_exports_a_representative_objective() -> None:
    compiled = qfin.compile(
        qfin.CVaR(
            qfin.LossDistribution([0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4]),
            confidence=0.75,
        ),
        target_error=0.2,
        min_qubits=2,
        max_qubits=2,
    )
    assert isinstance(compiled, qfin.CompiledRiskModel)
    exported = compiled.to_openqasm(power=0, target="all_to_all")
    assert "OPENQASM 2.0" in exported.program
    assert exported.target.wires == compiled.representation.qubits + 2
