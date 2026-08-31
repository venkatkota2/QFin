"""PennyLane runtime for factorized state preparation and arithmetic tail oracles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import CircuitObservation
from qfin.circuits import FactorizedPreparation, apply_zero_reflection
from qfin.exceptions import BackendUnavailableError, ResourceLimitError
from qfin.representation.arithmetic import StructuredLossOraclePlan
from qfin.representation.factorized import FactorizedDistributionEncoding


class FactorizedTailPennyLaneBackend:
    """Execute an MLAE tail indicator without a joint probability/payoff table."""

    def __init__(
        self,
        representation: FactorizedDistributionEncoding,
        oracle: StructuredLossOraclePlan,
        *,
        threshold: float,
        inclusive: bool,
        encoded_probability: float,
        device_name: str = "lightning.qubit",
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> None:
        if oracle.input_qubits != representation.qubits_per_factor:
            raise ValueError("oracle input registers do not match the factorized representation")
        if not 0 <= encoded_probability <= 1:
            raise ValueError("encoded_probability must lie in [0, 1]")
        if oracle.integer_monomials > max_integer_monomials:
            raise ResourceLimitError(
                f"structured arithmetic requires {oracle.integer_monomials} integer monomials, "
                f"above max_integer_monomials={max_integer_monomials}"
            )
        self.representation = representation
        self.oracle = oracle
        self.threshold = threshold
        self.inclusive = inclusive
        self.encoded_probability = encoded_probability
        self.device_name = device_name
        self.distribution_loader = FactorizedPreparation.from_encoding(representation)

        cursor = 0
        input_registers: list[tuple[int, ...]] = []
        for qubits in representation.qubits_per_factor:
            input_registers.append(tuple(range(cursor, cursor + qubits)))
            cursor += qubits
        self.input_registers = tuple(input_registers)
        self.data_wires = sum(self.input_registers, ())
        self.loss_wires = tuple(range(cursor, cursor + oracle.loss_qubits))
        cursor += oracle.loss_qubits
        affine_registers: list[tuple[int, ...]] = []
        for qubits in oracle.affine_qubits:
            affine_registers.append(tuple(range(cursor, cursor + qubits)))
            cursor += qubits
        self.affine_output_registers = tuple(affine_registers)
        self.piecewise_work_wire = cursor if oracle.piecewise_work_qubits else None
        cursor += oracle.piecewise_work_qubits
        self.objective_wire = cursor
        cursor += 1
        self.work_wire = cursor
        cursor += 1
        self.total_wires = cursor
        if self.total_wires > max_total_wires:
            raise ResourceLimitError(
                f"factorized tail circuit requires {self.total_wires} wires, "
                f"above max_total_wires={max_total_wires}"
            )
        self.register_wires = (
            *self.data_wires,
            *self.loss_wires,
            *sum(self.affine_output_registers, ()),
            self.objective_wire,
        )
        self._threshold_code = oracle.threshold_code(threshold, inclusive=inclusive)

    @staticmethod
    def _qml() -> Any:
        try:
            import pennylane as qml
        except ImportError as exc:
            raise BackendUnavailableError(
                "PennyLane is required to execute factorized quantum circuits. "
                "Install QFin with `python -m pip install -e '.[quantum]'`."
            ) from exc
        return qml

    @property
    def state_preparation_parameters(self) -> int:
        return self.distribution_loader.parameter_count

    @property
    def integer_monomials(self) -> int:
        return self.oracle.integer_monomials

    def theoretical_amplitude(self) -> float:
        return self.encoded_probability

    def _apply_distribution(self) -> None:
        self.distribution_loader.apply(self.data_wires)

    def _apply_indicator(self) -> None:
        qml = self._qml()
        if self._threshold_code <= 0:
            qml.PauliX(wires=self.objective_wire)
        elif self._threshold_code < 2**self.oracle.loss_qubits:
            qml.IntegerComparator(
                self._threshold_code,
                geq=True,
                wires=(*self.loss_wires, self.objective_wire),
            )

    def _apply_a(self) -> None:
        self._apply_distribution()
        self.oracle.apply(
            self.input_registers,
            self.loss_wires,
            self.affine_output_registers,
            self.piecewise_work_wire,
        )
        self._apply_indicator()

    def queue_circuit(self, power: int = 0) -> None:
        """Queue one factorized MLAE circuit on the active PennyLane tape."""

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
        values = np.asarray(self._make_circuit(power, shots=shots, seed=seed)())
        return float(values[1])

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
            probability = self.probability(
                power,
                shots=shots,
                seed=None if seed is None else seed + index,
            )
            observations.append(
                CircuitObservation(
                    power=power,
                    successes=int(np.clip(round(probability * shots), 0, shots)),
                    shots=shots,
                )
            )
        return tuple(observations)

    def circuit_specs(self, power: int = 0) -> dict[str, object]:
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

    def loss_probabilities(self) -> NDArray[np.float64]:
        """Execute state preparation plus arithmetic and measure the loss register."""

        qml = self._qml()
        device = qml.device(self.device_name, wires=self.total_wires)

        @qml.qnode(device)  # type: ignore[untyped-decorator]
        def circuit() -> Any:
            self._apply_distribution()
            self.oracle.apply(
                self.input_registers,
                self.loss_wires,
                self.affine_output_registers,
                self.piecewise_work_wire,
            )
            return qml.probs(wires=self.loss_wires)

        return np.asarray(circuit(), dtype=np.float64)


__all__ = ["FactorizedTailPennyLaneBackend"]
