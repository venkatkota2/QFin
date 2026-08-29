"""PennyLane runtime for quantum objectives over empirical financial losses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import CircuitObservation
from qfin.backends.structured import StructuredPennyLaneBackend
from qfin.representation import DistributionEncoding, QuantumObjectiveEncoding


class RiskPennyLaneBackend:
    """Execute risk objectives with QFin's gate-decomposable probability loader.

    The class reuses ``StructuredPennyLaneBackend`` for every objective. It is
    an orchestration adapter, not a simulator; PennyLane-Lightning continues to
    perform state-vector evolution and measurement.
    """

    def __init__(
        self,
        representation: DistributionEncoding,
        *,
        device_name: str = "lightning.qubit",
        max_structured_rotations: int = 32_767,
    ) -> None:
        if max_structured_rotations < 1:
            raise ValueError("max_structured_rotations must be positive")
        self.representation = representation
        self.device_name = device_name
        self.max_structured_rotations = max_structured_rotations
        self.total_wires = representation.qubits + 2

    def objective_backend(self, objective: QuantumObjectiveEncoding) -> StructuredPennyLaneBackend:
        """Build the existing structured runtime for one normalized objective."""

        if objective.distribution is not self.representation and (
            objective.distribution.qubits != self.representation.qubits
            or not np.array_equal(objective.distribution.grid, self.representation.grid)
            or not np.array_equal(
                objective.distribution.probabilities,
                self.representation.probabilities,
            )
        ):
            raise ValueError("objective distribution does not match this runtime")
        return StructuredPennyLaneBackend(
            self.representation,
            objective.normalized_values,
            device_name=self.device_name,
            max_structured_rotations=self.max_structured_rotations,
        )

    def theoretical_amplitude(self, objective: QuantumObjectiveEncoding) -> float:
        return self.objective_backend(objective).theoretical_amplitude()

    def probability(
        self,
        objective: QuantumObjectiveEncoding,
        power: int = 0,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        return self.objective_backend(objective).probability(power, shots=shots, seed=seed)

    def run_schedule(
        self,
        objective: QuantumObjectiveEncoding,
        schedule: Sequence[int],
        *,
        shots: int,
        seed: int | None = None,
    ) -> tuple[CircuitObservation, ...]:
        return self.objective_backend(objective).run_schedule(schedule, shots=shots, seed=seed)

    def distribution_probabilities(self) -> NDArray[np.float64]:
        """Execute only the empirical probability-tree state preparation."""

        objective = QuantumObjectiveEncoding(
            distribution=self.representation,
            normalized_values=np.zeros(self.representation.grid_points),
            financial_scale=0.0,
            financial_offset=0.0,
            label="distribution_state_inspection",
        )
        return self.objective_backend(objective).distribution_probabilities()

    def joint_state(self, objective: QuantumObjectiveEncoding) -> NDArray[np.complex128]:
        return self.objective_backend(objective).joint_state()

    def draw(self, objective: QuantumObjectiveEncoding, power: int = 0) -> str:
        return self.objective_backend(objective).draw(power)

    def circuit_specs(
        self, objective: QuantumObjectiveEncoding, power: int = 0
    ) -> dict[str, object]:
        return self.objective_backend(objective).circuit_specs(power)


__all__ = ["RiskPennyLaneBackend"]
