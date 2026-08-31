"""Resource reports for reversible factorized loss-oracle workflows."""

from __future__ import annotations

from dataclasses import dataclass

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


__all__ = ["StructuredFactorResourceReport", "StructuredTargetComparison"]
