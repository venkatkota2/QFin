"""Gate-decomposable PennyLane backend for QFin v0.2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import CircuitObservation
from qfin.backends.runtime import (
    ComplexPrecision,
    create_device,
    execute_probability_circuits,
    load_pennylane,
)
from qfin.circuits import (
    PayoffRotation,
    ProbabilityTreePreparation,
    apply_zero_reflection,
)
from qfin.exceptions import ResourceLimitError
from qfin.representation import DistributionEncoding


class StructuredPennyLaneBackend:
    """Execute QFin with multiplexed rotations and gate-level reflections.

    No dense ``QubitUnitary`` or diagonal matrix is created. Classical memory
    grows with the number of grid points rather than the square of the Hilbert
    space dimension.
    """

    def __init__(
        self,
        representation: DistributionEncoding,
        normalized_payoff: NDArray[np.float64],
        *,
        device_name: str = "lightning.qubit",
        precision: ComplexPrecision = "complex128",
        max_structured_rotations: int = 32_767,
    ) -> None:
        payoff = np.asarray(normalized_payoff, dtype=np.float64).reshape(-1)
        if payoff.shape != representation.probabilities.shape:
            raise ValueError("normalized_payoff must match the representation grid")
        self.representation = representation
        self.normalized_payoff = payoff
        self.device_name = device_name
        self.resolved_device_name = device_name
        self.precision = precision
        self.data_wires = tuple(range(representation.qubits))
        self.objective_wire = representation.qubits
        self.register_wires = (*self.data_wires, self.objective_wire)
        self.work_wire = representation.qubits + 1
        self.total_wires = representation.qubits + 2
        self.distribution_loader = ProbabilityTreePreparation.from_probabilities(
            representation.probabilities
        )
        self.payoff_loader = PayoffRotation.from_normalized_payoff(payoff)
        total_rotations = (
            self.distribution_loader.rotation_count + self.payoff_loader.rotation_count
        )
        if total_rotations > max_structured_rotations:
            raise ResourceLimitError(
                f"structured loading requires {total_rotations} conditional rotations, "
                f"above the configured limit of {max_structured_rotations}; "
                "reduce max_qubits or increase max_structured_rotations explicitly"
            )

    @property
    def structured_parameter_count(self) -> int:
        return (
            self.distribution_loader.parameter_count + self.payoff_loader.parameter_count
        )

    def theoretical_amplitude(self) -> float:
        return float(np.dot(self.representation.probabilities, self.normalized_payoff))

    def _apply_distribution(self) -> None:
        self.distribution_loader.apply(self.data_wires)

    def _apply_a(self) -> None:
        self._apply_distribution()
        self.payoff_loader.apply(self.data_wires, self.objective_wire)

    def _make_circuit(
        self,
        power: int,
        *,
        shots: int | None,
        seed: int | None,
        device: Any | None = None,
    ) -> Any:
        if power < 0:
            raise ValueError("power must be non-negative")
        qml = load_pennylane()
        if device is None:
            device, self.resolved_device_name = create_device(
                qml,
                self.device_name,
                wires=self.total_wires,
                seed=seed,
                precision=self.precision,
            )
        objective_wire = self.objective_wire
        register_wires = self.register_wires
        work_wire = self.work_wire

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            self._apply_a()
            for _ in range(power):
                # Q = -A S_0 A^† S_good. The global minus sign is irrelevant.
                qml.PauliZ(wires=objective_wire)
                qml.adjoint(self._apply_a)()
                apply_zero_reflection(register_wires, work_wire=work_wire)
                self._apply_a()
            return qml.probs(wires=objective_wire)

        if shots is not None:
            return qml.set_shots(circuit, shots=shots)
        return circuit

    def probability(
        self,
        power: int = 0,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        qml = load_pennylane()
        device, self.resolved_device_name = create_device(
            qml,
            self.device_name,
            wires=self.total_wires,
            seed=seed,
            precision=self.precision,
        )
        circuit = self._make_circuit(
            power, shots=shots, seed=seed, device=device
        )
        probabilities = execute_probability_circuits((circuit,))[0]
        return float(probabilities[1])

    def distribution_probabilities(self) -> NDArray[np.float64]:
        """Execute only the distribution loader and measure the data register."""
        qml = load_pennylane()
        device, self.resolved_device_name = create_device(
            qml,
            self.device_name,
            wires=self.total_wires,
            seed=None,
            precision=self.precision,
        )
        data_wires = self.data_wires

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            self._apply_distribution()
            return qml.probs(wires=data_wires)

        return np.asarray(circuit(), dtype=np.float64)

    def joint_state(self) -> NDArray[np.complex128]:
        """Return the exact simulator state after distribution and payoff loading."""
        qml = load_pennylane()
        device, self.resolved_device_name = create_device(
            qml,
            self.device_name,
            wires=self.total_wires,
            seed=None,
            precision=self.precision,
        )

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            self._apply_a()
            return qml.state()

        return np.asarray(circuit(), dtype=np.complex128)

    def run_schedule(
        self,
        schedule: Sequence[int],
        *,
        shots: int,
        seed: int | None = None,
    ) -> tuple[CircuitObservation, ...]:
        if shots <= 0:
            raise ValueError("shots must be positive")
        powers = tuple(int(power) for power in schedule)
        if not powers or len(set(powers)) != len(powers) or any(power < 0 for power in powers):
            raise ValueError("schedule must contain unique, non-negative powers")
        qml = load_pennylane()
        device, self.resolved_device_name = create_device(
            qml,
            self.device_name,
            wires=self.total_wires,
            seed=seed,
            precision=self.precision,
        )
        circuits = tuple(
            self._make_circuit(power, shots=shots, seed=seed, device=device)
            for power in powers
        )
        batch = execute_probability_circuits(circuits)
        observations: list[CircuitObservation] = []
        for power, probabilities in zip(powers, batch, strict=True):
            probability = float(probabilities[1])
            successes = int(np.clip(round(probability * shots), 0, shots))
            observations.append(
                CircuitObservation(power=power, successes=successes, shots=shots)
            )
        return tuple(observations)

    def draw(self, power: int = 0) -> str:
        qml = load_pennylane()
        circuit = self._make_circuit(power, shots=None, seed=None)
        return str(qml.draw(circuit)())

    def circuit_specs(self, power: int = 0) -> dict[str, object]:
        """Return PennyLane device-level circuit specifications."""
        qml = load_pennylane()
        circuit = self._make_circuit(power, shots=None, seed=None)
        specs = qml.specs(circuit, level="device")()
        resources = specs["resources"]
        return {
            "power": power,
            "num_wires": int(resources.num_allocs),
            "num_gates": int(resources.num_gates),
            "depth": int(resources.depth),
            "gate_types": dict(resources.gate_types),
        }
