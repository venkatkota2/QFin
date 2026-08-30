"""Quantile/Walsh PennyLane backend for QFin v0.3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import CircuitObservation
from qfin.circuits import (
    UniformQuantilePreparation,
    WalshPayoffApproximation,
    apply_zero_reflection,
)
from qfin.exceptions import BackendUnavailableError, ResourceLimitError
from qfin.representation import DistributionEncoding


class CompressedPennyLaneBackend:
    """Execute uniform quantile loading and a sparse Walsh payoff circuit."""

    def __init__(
        self,
        representation: DistributionEncoding,
        normalized_payoff: NDArray[np.float64],
        payoff_approximation: WalshPayoffApproximation,
        *,
        device_name: str = "lightning.qubit",
        max_compressed_terms: int = 32_767,
    ) -> None:
        payoff = np.asarray(normalized_payoff, dtype=np.float64).reshape(-1)
        if payoff.shape != representation.probabilities.shape:
            raise ValueError("normalized_payoff must match the representation grid")
        if np.any((payoff < 0) | (payoff > 1)) or not np.all(np.isfinite(payoff)):
            raise ValueError("normalized_payoff values must lie in [0, 1]")
        uniform = np.full(representation.grid_points, 1.0 / representation.grid_points)
        if not np.allclose(representation.probabilities, uniform, atol=1e-14):
            raise ValueError("compressed backend requires a uniform quantile encoding")
        if representation.state_preparation_method != "uniform_quantile_hadamard":
            raise ValueError("compressed backend requires uniform_quantile_hadamard")
        if payoff_approximation.qubits != representation.qubits:
            raise ValueError("payoff approximation must match representation qubits")
        if payoff_approximation.parameter_count > max_compressed_terms:
            raise ResourceLimitError(
                f"compressed payoff requires {payoff_approximation.parameter_count} "
                f"Pauli rotations, above the configured limit of {max_compressed_terms}; "
                "reduce max_qubits or increase max_compressed_terms explicitly"
            )
        self.representation = representation
        self.normalized_payoff = payoff
        self.payoff_loader = payoff_approximation
        self.distribution_loader = UniformQuantilePreparation(representation.qubits)
        self.device_name = device_name
        self.data_wires = tuple(range(representation.qubits))
        self.objective_wire = representation.qubits
        self.register_wires = (*self.data_wires, self.objective_wire)
        self.work_wire = representation.qubits + 1
        self.total_wires = representation.qubits + 2

    @staticmethod
    def _qml() -> Any:
        try:
            import pennylane as qml
        except ImportError as exc:
            raise BackendUnavailableError(
                "PennyLane is required to execute quantum circuits. "
                "Install QFin with `python -m pip install -e '.[quantum]'`."
            ) from exc
        return qml

    @property
    def structured_parameter_count(self) -> int:
        return self.payoff_loader.parameter_count

    def theoretical_amplitude(self) -> float:
        """Success amplitude of the compiled, possibly approximate circuit."""
        return self.payoff_loader.approximate_amplitude

    def exact_discrete_amplitude(self) -> float:
        """Success amplitude before payoff compression."""
        return float(np.dot(self.representation.probabilities, self.normalized_payoff))

    def _apply_distribution(self) -> None:
        self.distribution_loader.apply(self.data_wires)

    def _apply_a(self) -> None:
        self._apply_distribution()
        self.payoff_loader.apply(self.data_wires, self.objective_wire)

    def queue_circuit(self, power: int = 0) -> None:
        """Queue one decomposable MLAE circuit on the active PennyLane tape."""

        if power < 0:
            raise ValueError("power must be non-negative")
        qml = self._qml()
        self._apply_a()
        for _ in range(power):
            qml.PauliZ(wires=self.objective_wire)
            qml.adjoint(self._apply_a)()
            apply_zero_reflection(self.register_wires, work_wire=self.work_wire)
            self._apply_a()

    def circuit_tape(self, power: int = 0) -> Any:
        """Return a measurement-free tape for decomposition and export."""

        qml = self._qml()
        return qml.tape.make_qscript(lambda: self.queue_circuit(power))()

    def _make_circuit(self, power: int, *, shots: int | None, seed: int | None) -> Any:
        if power < 0:
            raise ValueError("power must be non-negative")
        qml = self._qml()
        device = qml.device(self.device_name, wires=self.total_wires, seed=seed)

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            self.queue_circuit(power)
            return qml.probs(wires=self.objective_wire)

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
        probabilities = np.asarray(self._make_circuit(power, shots=shots, seed=seed)())
        return float(probabilities[1])

    def distribution_probabilities(self) -> NDArray[np.float64]:
        """Execute only the parameter-free quantile loader."""
        qml = self._qml()
        device = qml.device(self.device_name, wires=self.total_wires)
        data_wires = self.data_wires

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            self._apply_distribution()
            return qml.probs(wires=data_wires)

        return np.asarray(circuit(), dtype=np.float64)

    def joint_state(self) -> NDArray[np.complex128]:
        """Return the simulator state after quantile and payoff loading."""
        qml = self._qml()
        device = qml.device(self.device_name, wires=self.total_wires)

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
        observations: list[CircuitObservation] = []
        for index, power in enumerate(powers):
            circuit_seed = None if seed is None else seed + index
            probability = self.probability(power, shots=shots, seed=circuit_seed)
            successes = int(np.clip(round(probability * shots), 0, shots))
            observations.append(CircuitObservation(power=power, successes=successes, shots=shots))
        return tuple(observations)

    def draw(self, power: int = 0) -> str:
        qml = self._qml()
        circuit = self._make_circuit(power, shots=None, seed=None)
        return str(qml.draw(circuit)())

    def circuit_specs(self, power: int = 0) -> dict[str, object]:
        """Return PennyLane device-level circuit specifications."""
        qml = self._qml()
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
