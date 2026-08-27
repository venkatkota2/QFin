"""PennyLane backend implementations and compatibility alias."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import CircuitObservation
from qfin.backends.compressed import CompressedPennyLaneBackend
from qfin.backends.structured import StructuredPennyLaneBackend
from qfin.exceptions import BackendUnavailableError, ResourceLimitError
from qfin.representation import DistributionEncoding

__all__ = [
    "CompressedPennyLaneBackend",
    "DensePennyLaneBackend",
    "PennyLaneBackend",
    "StructuredPennyLaneBackend",
]


class DensePennyLaneBackend:
    """Dense v0.1 reference backend retained for numerical validation.

    The adapter intentionally keeps dense Householder state preparation behind
    the backend boundary so a structured loader can replace it later without
    changing the finance or compiler APIs.
    """

    def __init__(
        self,
        representation: DistributionEncoding,
        normalized_payoff: NDArray[np.float64],
        *,
        device_name: str = "lightning.qubit",
        max_dense_dimension: int = 2_048,
    ) -> None:
        payoff = np.asarray(normalized_payoff, dtype=np.float64).reshape(-1)
        if payoff.shape != representation.probabilities.shape:
            raise ValueError("normalized_payoff must match the representation grid")
        if np.any((payoff < 0) | (payoff > 1)) or not np.all(np.isfinite(payoff)):
            raise ValueError("normalized_payoff values must lie in [0, 1]")
        self.representation = representation
        self.normalized_payoff = payoff
        self.device_name = device_name
        self.total_wires = representation.qubits + 1
        self.wires = tuple(range(self.total_wires))
        self.objective_wire = self.wires[-1]
        self._dimension = 2**self.total_wires
        if self._dimension > max_dense_dimension:
            raise ResourceLimitError(
                f"dense MVP state preparation needs a {self._dimension}x{self._dimension} "
                f"unitary, above the configured limit of {max_dense_dimension}; "
                "reduce max_qubits or provide a structured state-preparation backend"
            )
        self._joint_state = self._build_joint_state()
        self._preparation_unitary = self._householder(self._joint_state)
        self._zero_reflection = np.ones(self._dimension, dtype=np.complex128)
        self._zero_reflection[0] = -1.0

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

    def _build_joint_state(self) -> NDArray[np.float64]:
        probabilities = self.representation.probabilities
        state = np.zeros(2 * probabilities.size, dtype=np.float64)
        state[0::2] = np.sqrt(probabilities * (1.0 - self.normalized_payoff))
        state[1::2] = np.sqrt(probabilities * self.normalized_payoff)
        state /= np.linalg.norm(state)
        return state

    @staticmethod
    def _householder(state: NDArray[np.float64]) -> NDArray[np.complex128]:
        basis_zero = np.zeros_like(state)
        basis_zero[0] = 1.0
        difference = basis_zero - state
        norm = float(np.linalg.norm(difference))
        if norm < 1e-14:
            return np.eye(state.size, dtype=np.complex128)
        direction = difference / norm
        unitary = np.eye(state.size) - 2.0 * np.outer(direction, direction)
        return np.asarray(unitary, dtype=np.complex128)

    @property
    def joint_state(self) -> NDArray[np.float64]:
        """Copy of the probability-and-payoff state prepared by the circuit."""
        return self._joint_state.copy()

    @property
    def preparation_unitary(self) -> NDArray[np.complex128]:
        """Copy of the dense simulator-only state-preparation unitary."""
        return self._preparation_unitary.copy()

    def theoretical_amplitude(self) -> float:
        """Exact encoded objective amplitude before shot noise."""
        return float(np.dot(self.representation.probabilities, self.normalized_payoff))

    def _make_circuit(self, power: int, *, shots: int | None, seed: int | None) -> Any:
        if power < 0:
            raise ValueError("power must be non-negative")
        qml = self._qml()
        device = qml.device(
            self.device_name,
            wires=self.total_wires,
            seed=seed,
        )
        unitary = self._preparation_unitary
        inverse = unitary.conj().T
        zero_reflection = self._zero_reflection
        wires = self.wires
        objective_wire = self.objective_wire

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            qml.QubitUnitary(unitary, wires=wires, unitary_check=False)
            for _ in range(power):
                # Q = -A S_0 A^† S_good. The global minus sign is irrelevant.
                qml.PauliZ(wires=objective_wire)
                qml.QubitUnitary(inverse, wires=wires, unitary_check=False)
                qml.DiagonalQubitUnitary(zero_reflection, wires=wires)
                qml.QubitUnitary(unitary, wires=wires, unitary_check=False)
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
        """Execute one circuit and return objective-qubit success probability."""
        probabilities = np.asarray(self._make_circuit(power, shots=shots, seed=seed)())
        return float(probabilities[1])

    def run_schedule(
        self,
        schedule: Sequence[int],
        *,
        shots: int,
        seed: int | None = None,
    ) -> tuple[CircuitObservation, ...]:
        """Execute each Grover power and return binomial observations."""
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
            observations.append(
                CircuitObservation(power=power, successes=successes, shots=shots)
            )
        return tuple(observations)

    def draw(self, power: int = 0) -> str:
        """Render a text diagram of one exact-probability circuit."""
        qml = self._qml()
        circuit = self._make_circuit(power, shots=None, seed=None)
        return str(qml.draw(circuit)())
# Backward-compatible constructor; compiled models select the v0.3 default.
PennyLaneBackend = StructuredPennyLaneBackend
