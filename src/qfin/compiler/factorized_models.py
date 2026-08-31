"""Compiler policy and results for structured factorized tail objectives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from qfin.algorithms import AmplitudeEstimate, maximum_likelihood_amplitude_estimate
from qfin.backends.devices import DeviceTarget, resolve_quantum_device
from qfin.backends.factorized import FactorizedTailPennyLaneBackend
from qfin.backends.structured import StructuredPennyLaneBackend
from qfin.circuits import FactorizedPreparation
from qfin.exceptions import CompilationError, ResourceLimitError
from qfin.finance.exposures import (
    FactorTailProbability,
    FactorTailProbabilitySummary,
)
from qfin.representation.arithmetic import (
    StructuredLossOraclePlan,
    StructuredTailOracleValidation,
    compile_structured_loss_oracle,
    validate_structured_tail_oracle,
)
from qfin.representation.encoding import DistributionEncoding
from qfin.representation.strategies import (
    RepresentationTarget,
    StatePreparationStrategyReport,
    compare_state_preparation_strategies,
)
from qfin.resources.device import DeviceResourceReport, estimate_device_resources
from qfin.resources.structured import (
    StructuredFactorResourceReport,
    StructuredTargetComparison,
)


@dataclass(frozen=True, slots=True)
class StructuredOracleErrorBudget:
    """Probability-error allocation for a fixed-point comparator workflow."""

    total: float
    transform: float
    payoff: float
    estimation: float
    interval_level: float = 0.95

    @property
    def oracle(self) -> float:
        """Combined probability budget for transform and payoff quantization."""

        return self.transform + self.payoff

    @classmethod
    def allocate(cls, target_error: float) -> StructuredOracleErrorBudget:
        if not isfinite(target_error) or target_error <= 0:
            raise ValueError("target_error must be finite and greater than zero")
        return cls(
            total=target_error,
            transform=0.2 * target_error,
            payoff=0.2 * target_error,
            estimation=0.6 * target_error,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "total": self.total,
            "transform": self.transform,
            "payoff": self.payoff,
            "oracle": self.oracle,
            "estimation": self.estimation,
            "interval_level": self.interval_level,
            "reference_note": (
                "The encoded factor distribution is the validation reference. "
                "Transform and payoff allocations share one measured comparator-disagreement "
                "check. Continuous marginal truncation/discretization remains in encoding metadata."
            ),
        }


@dataclass(frozen=True, slots=True)
class FactorQuantumTailResult:
    """MLAE result for a reversible factorized tail comparator."""

    probability: float
    confidence_interval_95: tuple[float, float]
    exact_encoded_probability: float
    oracle_probability: float
    absolute_error: float
    oracle_error: float
    disagreement_probability: float
    estimation_error: float
    target_error: float
    meets_target_error: bool
    estimate: AmplitudeEstimate
    resources: StructuredFactorResourceReport
    backend: str
    algorithm: str = "factorized_reversible_comparator_mlae"

    def to_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "confidence_interval_95": list(self.confidence_interval_95),
            "exact_encoded_probability": self.exact_encoded_probability,
            "oracle_probability": self.oracle_probability,
            "absolute_error": self.absolute_error,
            "oracle_error": self.oracle_error,
            "disagreement_probability": self.disagreement_probability,
            "estimation_error": self.estimation_error,
            "target_error": self.target_error,
            "meets_target_error": self.meets_target_error,
            "estimate": self.estimate.to_dict(),
            "resources": self.resources.to_dict(),
            "backend": self.backend,
            "algorithm": self.algorithm,
            "caveat": (
                "Experimental simulator workflow. Arithmetic is reversible and avoids "
                "joint lookup tables, but decomposition can be deep and this is not a "
                "quantum-advantage or fault-tolerant resource claim."
            ),
        }


@dataclass(frozen=True, slots=True)
class CompiledFactorTailModel:
    """Compiled factorized tail problem with classical and PennyLane paths."""

    problem: FactorTailProbability
    oracle: StructuredLossOraclePlan
    validation: StructuredTailOracleValidation
    target_error: float
    error_budget: StructuredOracleErrorBudget
    state_preparation_strategy: StatePreparationStrategyReport
    validation_chunk_size: int
    backend_name: str = "pennylane"
    algorithm_name: str = "factorized_reversible_comparator_mlae"
    quantum_algorithm_available: bool = True

    @property
    def oracle_converged(self) -> bool:
        return self.validation.disagreement_probability <= self.error_budget.oracle

    @property
    def compilation_converged(self) -> bool:
        return self.oracle_converged and self.state_preparation_strategy.selected is not None

    def run(self) -> FactorTailProbabilitySummary:
        """Return the exact streamed encoded-distribution reference."""

        return FactorTailProbabilitySummary(
            probability=self.validation.exact_probability,
            threshold=self.problem.threshold,
            inclusive=self.problem.inclusive,
            evaluated_points=self.validation.evaluated_points,
            chunks=self.validation.chunks,
        )

    def to_pennylane(
        self,
        *,
        device_name: str = "auto",
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> FactorizedTailPennyLaneBackend:
        if self.backend_name != "pennylane":
            raise ValueError(f"compiled backend is {self.backend_name!r}, not 'pennylane'")
        return FactorizedTailPennyLaneBackend(
            self.problem.model.encoding,
            self.oracle,
            threshold=self.problem.threshold,
            inclusive=self.problem.inclusive,
            encoded_probability=self.validation.oracle_probability,
            device_name=resolve_quantum_device(device_name),
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )

    def resources(
        self,
        *,
        device_name: str = "auto",
    ) -> StructuredFactorResourceReport:
        encoding = self.problem.model.encoding
        loader = FactorizedPreparation.from_encoding(encoding)
        total_qubits = encoding.total_qubits + self.oracle.arithmetic_qubits + 2
        resolved_device = resolve_quantum_device(device_name, require_available=False)
        return StructuredFactorResourceReport(
            factor_registers=encoding.factor_count,
            data_qubits=encoding.total_qubits,
            marginal_grid_points=encoding.stored_marginal_points,
            joint_grid_points=encoding.joint_grid_points,
            state_preparation_parameters=loader.parameter_count,
            state_preparation_gates=loader.gate_count,
            loss_qubits=self.oracle.loss_qubits,
            affine_qubits=self.oracle.total_affine_qubits,
            arithmetic_work_qubits=self.oracle.piecewise_work_qubits,
            objective_qubits=1,
            reflection_work_qubits=1,
            total_qubits=total_qubits,
            integer_monomials=self.oracle.integer_monomials,
            generic_joint_state_parameters=encoding.joint_grid_points - 1,
            generic_joint_payoff_parameters=encoding.joint_grid_points,
            validation_points=self.validation.evaluated_points,
            validation_chunk_size=self.validation_chunk_size,
            backend=f"pennylane.{resolved_device}",
            algorithm=self.algorithm_name,
        )

    def run_quantum(
        self,
        *,
        schedule: Sequence[int] = (0, 1, 2),
        shots: int = 1_000,
        seed: int | None = 0,
        likelihood_grid_size: int = 131_073,
        device_name: str = "auto",
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> FactorQuantumTailResult:
        runtime = self.to_pennylane(
            device_name=device_name,
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )
        observations = runtime.run_schedule(schedule, shots=shots, seed=seed)
        estimate = maximum_likelihood_amplitude_estimate(
            observations,
            grid_size=likelihood_grid_size,
        )
        probability = estimate.amplitude
        absolute_error = abs(probability - self.validation.exact_probability)
        return FactorQuantumTailResult(
            probability=probability,
            confidence_interval_95=(estimate.lower_95, estimate.upper_95),
            exact_encoded_probability=self.validation.exact_probability,
            oracle_probability=self.validation.oracle_probability,
            absolute_error=absolute_error,
            oracle_error=self.validation.oracle_error,
            disagreement_probability=self.validation.disagreement_probability,
            estimation_error=abs(probability - self.validation.oracle_probability),
            target_error=self.target_error,
            meets_target_error=absolute_error <= self.target_error,
            estimate=estimate,
            resources=self.resources(device_name=device_name),
            backend=f"pennylane.{runtime.device_name}:factorized-arithmetic",
        )

    def device_resources(
        self,
        *,
        schedule: Sequence[int] = (0,),
        shots: int = 1_000,
        target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> DeviceResourceReport:
        runtime = self.to_pennylane(
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )
        return estimate_device_resources(
            runtime,
            schedule=schedule,
            shots=shots,
            target=target,
        )

    def target_comparison(
        self,
        *,
        schedule: Sequence[int] = (0,),
        shots: int = 1_000,
        target: Literal["all_to_all", "linear"] = "all_to_all",
        max_joint_points: int = 4_096,
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> StructuredTargetComparison:
        """Transpile structured and generic paths for a guarded small benchmark."""

        encoding = self.problem.model.encoding
        points = encoding.joint_grid_points
        if points > max_joint_points:
            raise ResourceLimitError(
                f"generic comparison would materialize {points} points, "
                f"above max_joint_points={max_joint_points}"
            )
        _, losses, probabilities = self.problem.model.chunk(0, points)
        payoff = np.asarray(
            losses >= self.problem.threshold
            if self.problem.inclusive
            else losses > self.problem.threshold,
            dtype=np.float64,
        )
        generic_representation = DistributionEncoding(
            grid=losses,
            probabilities=probabilities,
            qubits=encoding.total_qubits,
            lower_bound=float(np.min(losses)),
            upper_bound=float(np.max(losses)),
            tail_probability=0.0,
            discretization_error=0.0,
            mean_error=0.0,
            objective="factorized-tail-reference",
        )
        generic_runtime = StructuredPennyLaneBackend(
            generic_representation,
            payoff,
            device_name=resolve_quantum_device("auto"),
            max_structured_rotations=max(32_767, 2 * points),
        )
        structured_runtime = self.to_pennylane(
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )
        structured_report = estimate_device_resources(
            structured_runtime,
            schedule=schedule,
            shots=shots,
            target=target,
        )
        generic_report = estimate_device_resources(
            generic_runtime,
            schedule=schedule,
            shots=shots,
            target=target,
        )
        selected = self.state_preparation_strategy.require_selected()
        return StructuredTargetComparison(
            topology=target,
            structured=structured_report,
            generic=generic_report,
            structured_classical_parameters=(
                selected.classical_parameters + self.oracle.integer_monomials
            ),
            generic_classical_parameters=(2 * points - 1),
            structured_stored_values=(selected.stored_values + self.oracle.integer_monomials),
            generic_stored_values=(3 * points - 1),
            joint_points=points,
        )

    def explain(self) -> str:
        return (
            f"QFin compiled {self.problem.model.encoding.factor_count} factor registers "
            f"({self.problem.model.encoding.total_qubits} data qubits) into a "
            f"{self.oracle.loss_qubits}-qubit fixed-point loss register.\n"
            f"Integer monomials: {self.oracle.integer_monomials}; joint payoff table: not built.\n"
            f"Exact encoded tail probability: {self.validation.exact_probability:.12g}; "
            f"quantized oracle probability: {self.validation.oracle_probability:.12g}; "
            f"disagreement mass: {self.validation.disagreement_probability:.3e}.\n"
            f"Probability-error allocation: transform={self.error_budget.transform:.3e}, "
            f"payoff={self.error_budget.payoff:.3e}, "
            f"estimation={self.error_budget.estimation:.3e}.\n"
            f"Backend policy: {self.backend_name}. PennyLane-Lightning performs simulation; "
            "QFin constructs finance-specific arithmetic and compiler metadata."
        )


def compile_factor_tail_problem(
    problem: FactorTailProbability,
    *,
    target_error: float,
    backend: str,
    representation_target: RepresentationTarget | None,
    max_state_preparation_parameters: int,
    max_state_preparation_memory_bytes: int,
    arithmetic_scale: float | None,
    max_loss_qubits: int,
    max_affine_output_qubits: int,
    max_validation_points: int,
    validation_chunk_size: int,
    max_integer_monomials: int,
    max_total_wires: int,
) -> CompiledFactorTailModel:
    """Select fixed-point precision and a truthful classical/quantum backend."""

    if backend not in ("auto", "classical", "pennylane"):
        raise CompilationError(
            "factorized tail compilation supports backend='auto', 'classical', or 'pennylane'"
        )
    budget = StructuredOracleErrorBudget.allocate(target_error)
    if arithmetic_scale is not None and (
        not isfinite(arithmetic_scale) or arithmetic_scale <= 0
    ):
        raise ValueError("arithmetic_scale must be finite and positive")
    scales = (
        (arithmetic_scale,)
        if arithmetic_scale is not None
        else tuple(float(2**power) for power in range(0, 13))
    )
    selected_plan: StructuredLossOraclePlan | None = None
    selected_validation: StructuredTailOracleValidation | None = None
    last_resource_error: ResourceLimitError | None = None
    for scale in scales:
        try:
            candidate_plan = compile_structured_loss_oracle(
                problem.model.encoding,
                problem.model.objective,
                loss_scale=scale,
                max_loss_qubits=max_loss_qubits,
                max_affine_output_qubits=max_affine_output_qubits,
            )
        except ResourceLimitError as exc:
            last_resource_error = exc
            continue
        candidate_validation = validate_structured_tail_oracle(
            problem,
            candidate_plan,
            chunk_size=validation_chunk_size,
            max_points=max_validation_points,
        )
        selected_plan = candidate_plan
        selected_validation = candidate_validation
        if candidate_validation.disagreement_probability <= budget.oracle:
            break
    if selected_plan is None or selected_validation is None:
        if last_resource_error is not None:
            raise last_resource_error
        raise ResourceLimitError("no fixed-point arithmetic scale could be compiled")
    if selected_plan.integer_monomials > max_integer_monomials:
        raise ResourceLimitError(
            f"structured oracle needs {selected_plan.integer_monomials} integer monomials, "
            f"above max_integer_monomials={max_integer_monomials}"
        )

    strategy = compare_state_preparation_strategies(
        problem.model.encoding,
        target=representation_target,
        ancilla_qubits=selected_plan.arithmetic_qubits + 2,
        max_parameters=max_state_preparation_parameters,
        max_memory_bytes=max_state_preparation_memory_bytes,
    )
    total_wires = problem.model.encoding.total_qubits + selected_plan.arithmetic_qubits + 2
    within_runtime_width = total_wires <= max_total_wires
    oracle_converged = selected_validation.disagreement_probability <= budget.oracle
    if backend == "pennylane":
        strategy.require_selected()
        if not within_runtime_width:
            raise ResourceLimitError(
                f"factorized tail circuit requires {total_wires} wires, "
                f"above max_total_wires={max_total_wires}"
            )
        if not oracle_converged:
            raise ResourceLimitError(
                "fixed-point oracle disagreement exceeds its allocated probability error"
            )
        resolved_backend = "pennylane"
    elif backend == "classical":
        resolved_backend = "classical"
    else:
        resolved_backend = (
            "pennylane"
            if strategy.selected is not None and within_runtime_width and oracle_converged
            else "classical"
        )
    return CompiledFactorTailModel(
        problem=problem,
        oracle=selected_plan,
        validation=selected_validation,
        target_error=target_error,
        error_budget=budget,
        state_preparation_strategy=strategy,
        validation_chunk_size=validation_chunk_size,
        backend_name=resolved_backend,
    )


__all__ = [
    "CompiledFactorTailModel",
    "FactorQuantumTailResult",
    "StructuredOracleErrorBudget",
    "compile_factor_tail_problem",
]
