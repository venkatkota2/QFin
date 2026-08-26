"""Transparent logical-resource estimates for the MVP circuits."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

BackendMode = Literal["compressed", "structured", "dense"]


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """Logical estimate; not a hardware-transpiled gate count."""

    data_qubits: int
    objective_qubits: int
    work_qubits: int
    estimation_qubits: int
    total_logical_qubits: int
    grid_points: int
    schedule: tuple[int, ...]
    circuits: int
    shots_per_circuit: int
    total_shots: int
    oracle_queries: int
    estimated_state_preparation_rotations: int
    estimated_max_primitive_gates: int
    distribution_rotations: int
    distribution_gates: int
    payoff_rotations: int
    payoff_compression_ratio: float
    classical_parameter_count: int
    dense_unitary_entries: int
    backend: str
    backend_mode: BackendMode
    estimate_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "data_qubits": self.data_qubits,
            "objective_qubits": self.objective_qubits,
            "work_qubits": self.work_qubits,
            "estimation_qubits": self.estimation_qubits,
            "total_logical_qubits": self.total_logical_qubits,
            "grid_points": self.grid_points,
            "schedule": list(self.schedule),
            "circuits": self.circuits,
            "shots_per_circuit": self.shots_per_circuit,
            "total_shots": self.total_shots,
            "oracle_queries": self.oracle_queries,
            "estimated_state_preparation_rotations": self.estimated_state_preparation_rotations,
            "estimated_max_primitive_gates": self.estimated_max_primitive_gates,
            "distribution_rotations": self.distribution_rotations,
            "distribution_gates": self.distribution_gates,
            "payoff_rotations": self.payoff_rotations,
            "payoff_compression_ratio": self.payoff_compression_ratio,
            "classical_parameter_count": self.classical_parameter_count,
            "dense_unitary_entries": self.dense_unitary_entries,
            "backend": self.backend,
            "backend_mode": self.backend_mode,
            "estimate_kind": self.estimate_kind,
            "caveat": (
                "High-level logical estimate before device-specific decomposition, "
                "routing, and transpilation; not a hardware runtime prediction."
            ),
        }


def estimate_resources(
    data_qubits: int,
    *,
    schedule: Sequence[int] = (0, 1, 2, 4),
    shots: int = 1_000,
    backend: str = "pennylane.default.qubit",
    backend_mode: BackendMode = "compressed",
    payoff_terms: int | None = None,
) -> ResourceReport:
    """Estimate resources for a non-adaptive MLAE execution."""

    powers = tuple(int(power) for power in schedule)
    if not powers or any(power < 0 for power in powers):
        raise ValueError("schedule must contain non-negative Grover powers")
    if len(set(powers)) != len(powers):
        raise ValueError("schedule powers must be unique")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if data_qubits < 1:
        raise ValueError("data_qubits must be positive")
    if backend_mode not in ("compressed", "structured", "dense"):
        raise ValueError("backend_mode must be 'compressed', 'structured', or 'dense'")

    max_power = max(powers)
    oracle_queries = shots * sum(2 * power + 1 for power in powers)
    joint_dimension = 2 ** (data_qubits + 1)

    grid_points = 2**data_qubits
    if payoff_terms is not None and not 0 <= payoff_terms <= grid_points:
        raise ValueError("payoff_terms must lie between zero and the grid size")

    if backend_mode == "compressed":
        work_qubits = 1
        total_qubits = data_qubits + 2
        distribution_rotations = 0
        distribution_gates = data_qubits
        payoff_rotations = grid_points if payoff_terms is None else payoff_terms
        state_preparation_rotations = payoff_rotations
        state_preparation_gates = distribution_gates + payoff_rotations
        register_qubits = data_qubits + 1
        zero_reflection_operations = 2 * register_qubits + 3
        max_gates = (2 * max_power + 1) * state_preparation_gates
        max_gates += max_power * (zero_reflection_operations + 1)
        classical_parameter_count = payoff_rotations
        dense_unitary_entries = 0
        estimate_kind = "quantile_walsh_pauli_logical_estimate"
    elif backend_mode == "structured":
        work_qubits = 1
        total_qubits = data_qubits + 2
        distribution_rotations = grid_points - 1
        distribution_gates = distribution_rotations
        payoff_rotations = grid_points
        state_preparation_rotations = distribution_rotations + payoff_rotations
        register_qubits = data_qubits + 1
        zero_reflection_operations = 2 * register_qubits + 3
        max_gates = (2 * max_power + 1) * state_preparation_rotations
        max_gates += max_power * (zero_reflection_operations + 1)
        classical_parameter_count = state_preparation_rotations
        dense_unitary_entries = 0
        estimate_kind = "structured_multiplexed_rotation_logical_estimate"
    else:
        work_qubits = 0
        total_qubits = data_qubits + 1
        distribution_rotations = 0
        distribution_gates = 0
        payoff_rotations = 0
        state_preparation_rotations = max(0, 2 * joint_dimension - 2)
        reflection_overhead = 2 * max_power * total_qubits
        max_gates = (2 * max_power + 1) * state_preparation_rotations
        max_gates += reflection_overhead
        classical_parameter_count = joint_dimension
        dense_unitary_entries = joint_dimension**2
        estimate_kind = "dense_householder_reference_estimate"

    return ResourceReport(
        data_qubits=data_qubits,
        objective_qubits=1,
        work_qubits=work_qubits,
        estimation_qubits=0,
        total_logical_qubits=total_qubits,
        grid_points=grid_points,
        schedule=powers,
        circuits=len(powers),
        shots_per_circuit=shots,
        total_shots=shots * len(powers),
        oracle_queries=oracle_queries,
        estimated_state_preparation_rotations=state_preparation_rotations,
        estimated_max_primitive_gates=max_gates,
        distribution_rotations=distribution_rotations,
        distribution_gates=distribution_gates,
        payoff_rotations=payoff_rotations,
        payoff_compression_ratio=payoff_rotations / grid_points,
        classical_parameter_count=classical_parameter_count,
        dense_unitary_entries=dense_unitary_entries,
        backend=backend,
        backend_mode=backend_mode,
        estimate_kind=estimate_kind,
    )
