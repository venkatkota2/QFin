"""Resource reports for reversible factorized loss-oracle workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qfin.resources.device import DeviceResourceReport


@dataclass(frozen=True, slots=True)
class StructuredFactorResourceReport:
    """Logical resources before target decomposition and routing."""

    factor_registers: int
    data_qubits: int
    marginal_grid_points: int
    joint_grid_points: int
    state_preparation_parameters: int
    state_preparation_gates: int
    loss_qubits: int
    affine_qubits: int
    arithmetic_work_qubits: int
    objective_qubits: int
    reflection_work_qubits: int
    total_qubits: int
    integer_monomials: int
    generic_joint_state_parameters: int
    generic_joint_payoff_parameters: int
    validation_points: int
    validation_chunk_size: int
    backend: str
    algorithm: str

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_registers": self.factor_registers,
            "data_qubits": self.data_qubits,
            "marginal_grid_points": self.marginal_grid_points,
            "joint_grid_points": self.joint_grid_points,
            "state_preparation_parameters": self.state_preparation_parameters,
            "state_preparation_gates": self.state_preparation_gates,
            "loss_qubits": self.loss_qubits,
            "affine_qubits": self.affine_qubits,
            "arithmetic_work_qubits": self.arithmetic_work_qubits,
            "objective_qubits": self.objective_qubits,
            "reflection_work_qubits": self.reflection_work_qubits,
            "total_qubits": self.total_qubits,
            "integer_monomials": self.integer_monomials,
            "generic_joint_state_parameters": self.generic_joint_state_parameters,
            "generic_joint_payoff_parameters": self.generic_joint_payoff_parameters,
            "validation_points": self.validation_points,
            "validation_chunk_size": self.validation_chunk_size,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "joint_probability_table_materialized": False,
            "joint_payoff_table_materialized": False,
            "caveat": (
                "Logical arithmetic counts exclude fault-tolerant synthesis. Exact classical "
                "validation streams every encoded point and remains exponential in time."
            ),
        }


@dataclass(frozen=True, slots=True)
class StructuredTargetComparison:
    """Target-transpiled structured oracle versus a guarded generic reference."""

    topology: str
    structured: DeviceResourceReport
    generic: DeviceResourceReport
    structured_classical_parameters: int
    generic_classical_parameters: int
    structured_stored_values: int
    generic_stored_values: int
    joint_points: int
    generic_joint_materialized_for_benchmark: bool = True

    @property
    def routed_gate_ratio(self) -> float:
        return self.structured.total_routed_gates_per_objective / max(
            self.generic.total_routed_gates_per_objective, 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "structured": self.structured.to_dict(),
            "generic": self.generic.to_dict(),
            "structured_classical_parameters": self.structured_classical_parameters,
            "generic_classical_parameters": self.generic_classical_parameters,
            "structured_stored_values": self.structured_stored_values,
            "generic_stored_values": self.generic_stored_values,
            "joint_points": self.joint_points,
            "routed_gate_ratio": self.routed_gate_ratio,
            "generic_joint_materialized_for_benchmark": (
                self.generic_joint_materialized_for_benchmark
            ),
            "caveat": (
                "The two circuits use different ancillary widths. This is a measured "
                "portable decomposition/routing comparison, not a hardware runtime or "
                "quantum-advantage claim."
            ),
        }


@dataclass(frozen=True, slots=True)
class StructuredRiskResourceReport:
    """Logical workload for factorized VaR/CVaR hybrid MLAE execution."""

    base_oracle: StructuredFactorResourceReport
    problem_kind: Literal["value_at_risk", "conditional_value_at_risk"]
    schedule: tuple[int, ...]
    shots_per_circuit: int
    threshold_objectives: int
    excess_bit_objectives: int
    objective_evaluations: int
    total_circuit_executions: int
    total_shots: int
    oracle_queries: int
    tail_runtime_qubits: int
    excess_runtime_qubits: int
    maximum_runtime_qubits: int

    def to_dict(self) -> dict[str, object]:
        return {
            "base_oracle": self.base_oracle.to_dict(),
            "problem_kind": self.problem_kind,
            "schedule": list(self.schedule),
            "shots_per_circuit": self.shots_per_circuit,
            "threshold_objectives": self.threshold_objectives,
            "excess_bit_objectives": self.excess_bit_objectives,
            "objective_evaluations": self.objective_evaluations,
            "total_circuit_executions": self.total_circuit_executions,
            "total_shots": self.total_shots,
            "oracle_queries": self.oracle_queries,
            "tail_runtime_qubits": self.tail_runtime_qubits,
            "excess_runtime_qubits": self.excess_runtime_qubits,
            "maximum_runtime_qubits": self.maximum_runtime_qubits,
            "joint_probability_table_materialized": False,
            "joint_payoff_table_materialized": False,
            "caveat": (
                "Counts include every hybrid threshold and excess-bit objective. "
                "They are logical simulator-workload estimates, not fault-tolerant costs."
            ),
        }


__all__ = [
    "StructuredFactorResourceReport",
    "StructuredRiskResourceReport",
    "StructuredTargetComparison",
]
