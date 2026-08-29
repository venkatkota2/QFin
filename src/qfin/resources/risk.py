"""Logical resource and classical preprocessing estimates for quantum risk."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, log2
from typing import Literal

from qfin.resources.estimation import ResourceReport, estimate_resources

RiskProblemKind = Literal["tail_probability", "value_at_risk", "conditional_value_at_risk"]


@dataclass(frozen=True, slots=True)
class RiskResourceReport:
    """Aggregate logical resources for a hybrid quantum-risk workflow."""

    problem_kind: RiskProblemKind
    per_objective: ResourceReport
    threshold_evaluations: int
    excess_evaluations: int
    objective_evaluations: int
    total_circuits: int
    total_shots: int
    total_oracle_queries: int
    estimated_compiled_circuit_gates: int
    classical_input_points: int
    encoded_grid_points: int
    estimated_sort_comparisons: int
    estimated_preprocessing_bytes: int
    state_preparation_complexity: str
    threshold_search_complexity: str
    estimate_kind: str = "hybrid_risk_logical_upper_bound"

    def to_dict(self) -> dict[str, object]:
        return {
            "problem_kind": self.problem_kind,
            "per_objective": self.per_objective.to_dict(),
            "threshold_evaluations": self.threshold_evaluations,
            "excess_evaluations": self.excess_evaluations,
            "objective_evaluations": self.objective_evaluations,
            "total_circuits": self.total_circuits,
            "total_shots": self.total_shots,
            "total_oracle_queries": self.total_oracle_queries,
            "estimated_compiled_circuit_gates": (self.estimated_compiled_circuit_gates),
            "classical_input_points": self.classical_input_points,
            "encoded_grid_points": self.encoded_grid_points,
            "estimated_sort_comparisons": self.estimated_sort_comparisons,
            "estimated_preprocessing_bytes": self.estimated_preprocessing_bytes,
            "state_preparation_complexity": self.state_preparation_complexity,
            "threshold_search_complexity": self.threshold_search_complexity,
            "backend": self.per_objective.backend,
            "estimate_kind": self.estimate_kind,
            "caveat": (
                "Logical pre-transpilation estimate. The generic empirical loader "
                "and objective multiplexers scale exponentially with data qubits; "
                "this report is not a hardware runtime or quantum-advantage claim."
            ),
        }


def estimate_risk_resources(
    data_qubits: int,
    *,
    input_points: int,
    occupied_grid_points: int,
    problem_kind: RiskProblemKind,
    schedule: Sequence[int] = (0, 1, 2, 4),
    shots: int = 1_000,
    backend: str = "pennylane.default.qubit",
    threshold_evaluations: int | None = None,
) -> RiskResourceReport:
    """Estimate state loading, oracle, search, and preprocessing resources."""

    if input_points < 1:
        raise ValueError("input_points must be positive")
    if not 1 <= occupied_grid_points <= 2**data_qubits:
        raise ValueError("occupied_grid_points must lie in the encoded grid")
    if problem_kind not in (
        "tail_probability",
        "value_at_risk",
        "conditional_value_at_risk",
    ):
        raise ValueError("unsupported risk problem kind")
    if threshold_evaluations is None:
        if problem_kind == "tail_probability":
            resolved_threshold_evaluations = 1
        else:
            resolved_threshold_evaluations = max(1, ceil(log2(occupied_grid_points)) + 1)
    else:
        if threshold_evaluations < 1:
            raise ValueError("threshold_evaluations must be positive")
        resolved_threshold_evaluations = threshold_evaluations
    excess_evaluations = 1 if problem_kind == "conditional_value_at_risk" else 0
    objective_evaluations = resolved_threshold_evaluations + excess_evaluations
    per_objective = estimate_resources(
        data_qubits,
        schedule=schedule,
        shots=shots,
        backend=backend,
        backend_mode="structured",
    )
    encoded_grid_points = 2**data_qubits
    estimated_sort_comparisons = 0 if input_points < 2 else ceil(input_points * log2(input_points))
    # Input losses/probabilities plus encoded grid/probabilities, all doubles.
    estimated_preprocessing_bytes = 16 * (input_points + encoded_grid_points)
    return RiskResourceReport(
        problem_kind=problem_kind,
        per_objective=per_objective,
        threshold_evaluations=resolved_threshold_evaluations,
        excess_evaluations=excess_evaluations,
        objective_evaluations=objective_evaluations,
        total_circuits=objective_evaluations * per_objective.circuits,
        total_shots=objective_evaluations * per_objective.total_shots,
        total_oracle_queries=objective_evaluations * per_objective.oracle_queries,
        estimated_compiled_circuit_gates=(
            objective_evaluations * per_objective.estimated_max_primitive_gates
        ),
        classical_input_points=input_points,
        encoded_grid_points=encoded_grid_points,
        estimated_sort_comparisons=estimated_sort_comparisons,
        estimated_preprocessing_bytes=estimated_preprocessing_bytes,
        state_preparation_complexity=(
            "O(2**data_qubits) generic probability-tree and objective rotations"
        ),
        threshold_search_complexity=(
            "one fixed threshold"
            if problem_kind == "tail_probability"
            else "O(log(occupied_grid_points)) hybrid binary search"
        ),
    )


__all__ = ["RiskProblemKind", "RiskResourceReport", "estimate_risk_resources"]
