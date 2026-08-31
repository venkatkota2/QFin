"""Structured factorized VaR/CVaR compilation and hybrid MLAE execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite, log2
from typing import Literal

import numpy as np

from qfin.algorithms import AmplitudeEstimate, maximum_likelihood_amplitude_estimate
from qfin.backends.devices import DeviceTarget, resolve_quantum_device
from qfin.backends.factorized import (
    FactorizedExcessPennyLaneBackend,
    FactorizedTailPennyLaneBackend,
)
from qfin.circuits import FactorizedPreparation
from qfin.exceptions import CompilationError, ResourceLimitError
from qfin.finance.exposures import (
    FactorCVaR,
    FactorRiskProblem,
    FactorRiskSummary,
    FactorVaR,
    evaluate_factor_risk,
)
from qfin.representation.arithmetic import (
    StructuredLossOraclePlan,
    StructuredRiskOracleValidation,
    compile_structured_loss_oracle,
    validate_structured_risk_oracle,
)
from qfin.representation.strategies import (
    RepresentationTarget,
    StatePreparationStrategyReport,
    compare_state_preparation_strategies,
)
from qfin.resources.device import DeviceResourceReport, estimate_device_resources
from qfin.resources.structured import (
    StructuredFactorResourceReport,
    StructuredRiskResourceReport,
)

FactorRiskKind = Literal["value_at_risk", "conditional_value_at_risk"]


@dataclass(frozen=True, slots=True)
class StructuredRiskErrorBudget:
    """Financial-unit error allocation for structured VaR/CVaR."""

    total: float
    loss_quantization: float
    estimation: float
    interval_level: float = 0.95

    @classmethod
    def allocate(cls, target_error: float) -> StructuredRiskErrorBudget:
        if not isfinite(target_error) or target_error <= 0:
            raise ValueError("target_error must be finite and greater than zero")
        return cls(
            total=target_error,
            loss_quantization=0.4 * target_error,
            estimation=0.6 * target_error,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "total": self.total,
            "loss_quantization": self.loss_quantization,
            "estimation": self.estimation,
            "interval_level": self.interval_level,
            "reference_note": (
                "The exact encoded factor grid is streamed as the classical reference. "
                "Loss quantization is measured in financial units; stochastic search and "
                "MLAE error are reported against the selected encoded loss register."
            ),
        }


@dataclass(frozen=True, slots=True)
class FactorQuantumObjectiveEstimate:
    """One structured CDF or excess-register-bit MLAE estimate."""

    objective: Literal["cdf", "excess_bit"]
    encoded_probability: float
    estimate: AmplitudeEstimate
    threshold_code: int
    threshold: float
    complement: bool = False
    bit_index: int | None = None
    bit_weight: int | None = None

    @property
    def probability(self) -> float:
        return 1.0 - self.estimate.amplitude if self.complement else self.estimate.amplitude

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        if self.complement:
            return 1.0 - self.estimate.upper_95, 1.0 - self.estimate.lower_95
        return self.estimate.lower_95, self.estimate.upper_95

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "encoded_probability": self.encoded_probability,
            "probability": self.probability,
            "confidence_interval_95": list(self.confidence_interval_95),
            "threshold_code": self.threshold_code,
            "threshold": self.threshold,
            "complement": self.complement,
            "bit_index": self.bit_index,
            "bit_weight": self.bit_weight,
            "estimate": self.estimate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FactorQuantumVaRSearch:
    """Hybrid binary search over occupied structured loss codes."""

    confidence: float
    selected_code: int
    value: float
    confidence_interval_95: tuple[float, float]
    encoded_value: float
    evaluations: tuple[FactorQuantumObjectiveEstimate, ...]
    selected_estimate: FactorQuantumObjectiveEstimate
    resolved_to_single_loss_code: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "selected_code": self.selected_code,
            "value": self.value,
            "confidence_interval_95": list(self.confidence_interval_95),
            "encoded_value": self.encoded_value,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "selected_estimate": self.selected_estimate.to_dict(),
            "resolved_to_single_loss_code": self.resolved_to_single_loss_code,
            "caveat": (
                "The interval combines local MLAE intervals using CDF monotonicity; "
                "it is not a simultaneous-coverage guarantee."
            ),
        }


@dataclass(frozen=True, slots=True)
class FactorQuantumRiskResult:
    """Structured quantum VaR/CVaR estimate with explicit reference errors."""

    problem_kind: FactorRiskKind
    value: float
    confidence_interval_95: tuple[float, float]
    classical_value: float
    encoded_value: float
    absolute_error: float
    oracle_error: float
    estimation_error: float
    target_error: float
    meets_target_error: bool
    value_at_risk: float
    value_at_risk_interval_95: tuple[float, float]
    expected_shortfall: float | None
    search: FactorQuantumVaRSearch
    excess_estimates: tuple[FactorQuantumObjectiveEstimate, ...]
    resources: StructuredRiskResourceReport
    backend: str
    algorithm: str

    def to_dict(self) -> dict[str, object]:
        return {
            "problem_kind": self.problem_kind,
            "value": self.value,
            "confidence_interval_95": list(self.confidence_interval_95),
            "classical_value": self.classical_value,
            "encoded_value": self.encoded_value,
            "absolute_error": self.absolute_error,
            "oracle_error": self.oracle_error,
            "estimation_error": self.estimation_error,
            "target_error": self.target_error,
            "meets_target_error": self.meets_target_error,
            "value_at_risk": self.value_at_risk,
            "value_at_risk_interval_95": list(self.value_at_risk_interval_95),
            "expected_shortfall": self.expected_shortfall,
            "search": self.search.to_dict(),
            "excess_estimates": [item.to_dict() for item in self.excess_estimates],
            "resources": self.resources.to_dict(),
            "backend": self.backend,
            "algorithm": self.algorithm,
            "caveat": (
                "Experimental simulator workflow with reversible financial arithmetic. "
                "CVaR bounds are conditional on the selected VaR code and combine marginal "
                "bit intervals; no quantum-advantage or fault-tolerant claim is made."
            ),
        }


@dataclass(frozen=True, slots=True)
class CompiledFactorRiskModel:
    """Compiled factorized VaR/CVaR with streamed and PennyLane paths."""

    problem: FactorRiskProblem
    oracle: StructuredLossOraclePlan
    validation: StructuredRiskOracleValidation
    classical_summary: FactorRiskSummary
    target_error: float
    error_budget: StructuredRiskErrorBudget
    state_preparation_strategy: StatePreparationStrategyReport
    validation_chunk_size: int
    backend_name: str = "pennylane"
    algorithm_name: str = "factorized_var_search_and_bitwise_tail_excess_mlae"
    quantum_algorithm_available: bool = True

    @property
    def problem_kind(self) -> FactorRiskKind:
        return (
            "value_at_risk" if isinstance(self.problem, FactorVaR) else "conditional_value_at_risk"
        )

    @property
    def classical_value(self) -> float:
        return (
            self.classical_summary.var
            if isinstance(self.problem, FactorVaR)
            else self.classical_summary.cvar
        )

    @property
    def encoded_value(self) -> float:
        return (
            self.validation.oracle_value_at_risk
            if isinstance(self.problem, FactorVaR)
            else self.validation.oracle_expected_shortfall
        )

    @property
    def oracle_error(self) -> float:
        return abs(self.encoded_value - self.classical_value)

    @property
    def oracle_converged(self) -> bool:
        return self.oracle_error <= self.error_budget.loss_quantization

    @property
    def compilation_converged(self) -> bool:
        return self.oracle_converged and self.state_preparation_strategy.selected is not None

    def run(self) -> FactorRiskSummary:
        """Return the exact memory-bounded encoded-factor reference."""

        return self.classical_summary

    def _base_resources(self, *, device_name: str) -> StructuredFactorResourceReport:
        encoding = self.problem.model.encoding
        loader = FactorizedPreparation.from_encoding(encoding)
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
            total_qubits=encoding.total_qubits + self.oracle.arithmetic_qubits + 2,
            integer_monomials=self.oracle.integer_monomials,
            generic_joint_state_parameters=encoding.joint_grid_points - 1,
            generic_joint_payoff_parameters=encoding.joint_grid_points,
            validation_points=self.validation.evaluated_points,
            validation_chunk_size=self.validation_chunk_size,
            backend=f"pennylane.{resolved_device}",
            algorithm=self.algorithm_name,
        )

    def resources(
        self,
        *,
        schedule: Sequence[int] = (0, 1, 2),
        shots: int = 1_000,
        threshold_evaluations: int | None = None,
        device_name: str = "auto",
    ) -> StructuredRiskResourceReport:
        powers = tuple(int(power) for power in schedule)
        if not powers or len(set(powers)) != len(powers) or any(power < 0 for power in powers):
            raise ValueError("schedule must contain unique, non-negative powers")
        if shots <= 0:
            raise ValueError("shots must be positive")
        candidates = len(self.validation.occupied_codes)
        default_thresholds = max(1, ceil(log2(max(candidates, 1))) + 1)
        thresholds = default_thresholds if threshold_evaluations is None else threshold_evaluations
        if thresholds < 1:
            raise ValueError("threshold_evaluations must be positive")
        excess_bits = self.oracle.loss_qubits if isinstance(self.problem, FactorCVaR) else 0
        objectives = thresholds + excess_bits
        circuit_executions = objectives * len(powers)
        query_weight = sum(2 * power + 1 for power in powers)
        tail_qubits = self.problem.model.encoding.total_qubits + self.oracle.arithmetic_qubits + 2
        excess_qubits = (
            self.problem.model.encoding.total_qubits
            + self.oracle.arithmetic_qubits
            + self.oracle.loss_qubits
            + 2
        )
        return StructuredRiskResourceReport(
            base_oracle=self._base_resources(device_name=device_name),
            problem_kind=self.problem_kind,
            schedule=powers,
            shots_per_circuit=shots,
            threshold_objectives=thresholds,
            excess_bit_objectives=excess_bits,
            objective_evaluations=objectives,
            total_circuit_executions=circuit_executions,
            total_shots=circuit_executions * shots,
            oracle_queries=objectives * shots * query_weight,
            tail_runtime_qubits=tail_qubits,
            excess_runtime_qubits=excess_qubits if excess_bits else 0,
            maximum_runtime_qubits=max(tail_qubits, excess_qubits if excess_bits else 0),
        )

    def _tail_runtime(
        self,
        threshold_code: int,
        *,
        device_name: str,
        max_integer_monomials: int,
        max_total_wires: int,
    ) -> FactorizedTailPennyLaneBackend:
        if self.backend_name != "pennylane":
            raise ValueError(f"compiled backend is {self.backend_name!r}, not 'pennylane'")
        return FactorizedTailPennyLaneBackend(
            self.problem.model.encoding,
            self.oracle,
            threshold=float(
                self.oracle.decode_loss(np.asarray([threshold_code], dtype=np.int64))[0]
            ),
            inclusive=True,
            threshold_code=threshold_code,
            encoded_probability=self.validation.tail_at_code(threshold_code),
            device_name=resolve_quantum_device(device_name),
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )

    def tail_runtime(
        self,
        threshold_code: int,
        *,
        device_name: str = "auto",
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> FactorizedTailPennyLaneBackend:
        """Build a reusable loss-register threshold objective for inspection/testing."""

        return self._tail_runtime(
            threshold_code,
            device_name=device_name,
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )

    def excess_runtime(
        self,
        threshold_code: int,
        bit_index: int,
        *,
        device_name: str = "auto",
        max_integer_monomials: int = 4_096,
        max_total_wires: int = 28,
    ) -> FactorizedExcessPennyLaneBackend:
        """Build one positive-tail-excess bit objective for inspection/testing."""

        if self.backend_name != "pennylane":
            raise ValueError(f"compiled backend is {self.backend_name!r}, not 'pennylane'")
        probabilities = self.validation.excess_bit_probabilities(threshold_code)
        return FactorizedExcessPennyLaneBackend(
            self.problem.model.encoding,
            self.oracle,
            threshold_code=threshold_code,
            bit_index=bit_index,
            encoded_probability=probabilities[bit_index],
            device_name=resolve_quantum_device(device_name),
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )

    @staticmethod
    def _estimate(
        runtime: FactorizedTailPennyLaneBackend | FactorizedExcessPennyLaneBackend,
        *,
        schedule: tuple[int, ...],
        shots: int,
        seed: int | None,
        likelihood_grid_size: int,
    ) -> AmplitudeEstimate:
        observations = runtime.run_schedule(schedule, shots=shots, seed=seed)
        return maximum_likelihood_amplitude_estimate(
            observations,
            grid_size=likelihood_grid_size,
        )

    def _run_var_search(
        self,
        *,
        schedule: tuple[int, ...],
        shots: int,
        seed: int | None,
        likelihood_grid_size: int,
        device_name: str,
        max_integer_monomials: int,
        max_total_wires: int,
    ) -> FactorQuantumVaRSearch:
        candidates = self.validation.occupied_codes
        if not candidates:
            raise ValueError("structured loss register has no occupied codes")
        lower_index = 0
        upper_index = len(candidates) - 1
        evaluations: list[FactorQuantumObjectiveEstimate] = []
        by_index: dict[int, FactorQuantumObjectiveEstimate] = {}

        def evaluate(index: int) -> FactorQuantumObjectiveEstimate:
            candidate = candidates[index]
            tail_code = candidate + 1
            runtime = self._tail_runtime(
                tail_code,
                device_name=device_name,
                max_integer_monomials=max_integer_monomials,
                max_total_wires=max_total_wires,
            )
            estimate = self._estimate(
                runtime,
                schedule=schedule,
                shots=shots,
                seed=None if seed is None else seed + len(evaluations),
                likelihood_grid_size=likelihood_grid_size,
            )
            item = FactorQuantumObjectiveEstimate(
                objective="cdf",
                encoded_probability=self.validation.cdf_at_code(candidate),
                estimate=estimate,
                threshold_code=candidate,
                threshold=float(
                    self.oracle.decode_loss(np.asarray([candidate], dtype=np.int64))[0]
                ),
                complement=True,
            )
            evaluations.append(item)
            by_index[index] = item
            return item

        while lower_index < upper_index:
            middle = (lower_index + upper_index) // 2
            item = evaluate(middle)
            if item.probability >= self.problem.confidence:
                upper_index = middle
            else:
                lower_index = middle + 1
        selected_index = lower_index
        selected = by_index.get(selected_index) or evaluate(selected_index)

        interval_lower = 0
        interval_upper = len(candidates) - 1
        for index, item in by_index.items():
            lower, upper = item.confidence_interval_95
            if upper < self.problem.confidence:
                interval_lower = max(interval_lower, min(index + 1, len(candidates) - 1))
            if lower >= self.problem.confidence:
                interval_upper = min(interval_upper, index)
        if interval_lower > interval_upper:
            interval_lower = selected_index
            interval_upper = selected_index
        selected_code = candidates[selected_index]
        return FactorQuantumVaRSearch(
            confidence=self.problem.confidence,
            selected_code=selected_code,
            value=float(self.oracle.decode_loss(np.asarray([selected_code], dtype=np.int64))[0]),
            confidence_interval_95=(
                float(
                    self.oracle.decode_loss(
                        np.asarray([candidates[interval_lower]], dtype=np.int64)
                    )[0]
                ),
                float(
                    self.oracle.decode_loss(
                        np.asarray([candidates[interval_upper]], dtype=np.int64)
                    )[0]
                ),
            ),
            encoded_value=self.validation.oracle_value_at_risk,
            evaluations=tuple(evaluations),
            selected_estimate=selected,
            resolved_to_single_loss_code=interval_lower == interval_upper,
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
    ) -> FactorQuantumRiskResult:
        """Execute structured VaR search and, for CVaR, bitwise tail excess."""

        powers = tuple(int(power) for power in schedule)
        if not powers or len(set(powers)) != len(powers) or any(power < 0 for power in powers):
            raise ValueError("schedule must contain unique, non-negative powers")
        if shots <= 0:
            raise ValueError("shots must be positive")
        search = self._run_var_search(
            schedule=powers,
            shots=shots,
            seed=seed,
            likelihood_grid_size=likelihood_grid_size,
            device_name=device_name,
            max_integer_monomials=max_integer_monomials,
            max_total_wires=max_total_wires,
        )
        if isinstance(self.problem, FactorVaR):
            value = search.value
            interval = search.confidence_interval_95
            excess_estimates: tuple[FactorQuantumObjectiveEstimate, ...] = ()
            expected_shortfall = None
        else:
            bit_probabilities = self.validation.excess_bit_probabilities(search.selected_code)
            items: list[FactorQuantumObjectiveEstimate] = []
            expected_ticks = 0.0
            lower_ticks = 0.0
            upper_ticks = 0.0
            for bit_index, encoded_probability in enumerate(bit_probabilities):
                runtime = self.excess_runtime(
                    search.selected_code,
                    bit_index,
                    device_name=device_name,
                    max_integer_monomials=max_integer_monomials,
                    max_total_wires=max_total_wires,
                )
                estimate = self._estimate(
                    runtime,
                    schedule=powers,
                    shots=shots,
                    seed=(
                        None
                        if seed is None
                        else seed + len(search.evaluations) + bit_index + 10_000
                    ),
                    likelihood_grid_size=likelihood_grid_size,
                )
                weight = 1 << (self.oracle.loss_qubits - 1 - bit_index)
                item = FactorQuantumObjectiveEstimate(
                    objective="excess_bit",
                    encoded_probability=encoded_probability,
                    estimate=estimate,
                    threshold_code=search.selected_code,
                    threshold=search.value,
                    bit_index=bit_index,
                    bit_weight=weight,
                )
                items.append(item)
                expected_ticks += weight * item.probability
                bit_lower, bit_upper = item.confidence_interval_95
                lower_ticks += weight * bit_lower
                upper_ticks += weight * bit_upper
            multiplier = 1.0 / self.oracle.loss_scale / (1.0 - self.problem.confidence)
            value = search.value + multiplier * expected_ticks
            interval = (
                search.value + multiplier * lower_ticks,
                search.value + multiplier * upper_ticks,
            )
            excess_estimates = tuple(items)
            expected_shortfall = value

        resources = self.resources(
            schedule=powers,
            shots=shots,
            threshold_evaluations=len(search.evaluations),
            device_name=device_name,
        )
        return FactorQuantumRiskResult(
            problem_kind=self.problem_kind,
            value=value,
            confidence_interval_95=(min(interval), max(interval)),
            classical_value=self.classical_value,
            encoded_value=self.encoded_value,
            absolute_error=abs(value - self.classical_value),
            oracle_error=self.oracle_error,
            estimation_error=abs(value - self.encoded_value),
            target_error=self.target_error,
            meets_target_error=abs(value - self.classical_value) <= self.target_error,
            value_at_risk=search.value,
            value_at_risk_interval_95=search.confidence_interval_95,
            expected_shortfall=expected_shortfall,
            search=search,
            excess_estimates=excess_estimates,
            resources=resources,
            backend=f"pennylane.{resolve_quantum_device(device_name)}:factorized-arithmetic",
            algorithm=self.algorithm_name,
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
        """Conservatively profile the widest repeated structured objective."""

        logical = self.resources(schedule=schedule, shots=shots)
        representative_code = self.validation.occupied_codes[
            len(self.validation.occupied_codes) // 2
        ]
        runtime: FactorizedTailPennyLaneBackend | FactorizedExcessPennyLaneBackend
        if isinstance(self.problem, FactorCVaR):
            runtime = self.excess_runtime(
                representative_code,
                0,
                max_integer_monomials=max_integer_monomials,
                max_total_wires=max_total_wires,
            )
        else:
            runtime = self._tail_runtime(
                representative_code + 1,
                device_name="auto",
                max_integer_monomials=max_integer_monomials,
                max_total_wires=max_total_wires,
            )
        return estimate_device_resources(
            runtime,
            schedule=schedule,
            shots=shots,
            target=target,
            objective_evaluations=logical.objective_evaluations,
        )

    def explain(self) -> str:
        return (
            f"QFin compiled structured {self.problem_kind} over "
            f"{self.problem.model.encoding.factor_count} factor registers and a "
            f"{self.oracle.loss_qubits}-qubit fixed-point loss register.\n"
            f"Occupied loss codes: {len(self.validation.occupied_codes)}; "
            "joint probability/payoff table: not built.\n"
            f"Exact encoded-grid VaR/CVaR: {self.classical_summary.var:.12g} / "
            f"{self.classical_summary.cvar:.12g}; fixed-point VaR/CVaR: "
            f"{self.validation.oracle_value_at_risk:.12g} / "
            f"{self.validation.oracle_expected_shortfall:.12g}.\n"
            "Algorithm: hybrid MLAE CDF search; CVaR additionally computes a reversible "
            "positive tail-excess register and estimates its bits.\n"
            "PennyLane-Lightning performs simulation; QFin owns only the finance-specific "
            "representation, arithmetic, validation, and compiler workflow."
        )


def compile_factor_risk_problem(
    problem: FactorRiskProblem,
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
) -> CompiledFactorRiskModel:
    """Compile one factorized VaR/CVaR problem with measured precision selection."""

    if backend not in ("auto", "classical", "pennylane"):
        raise CompilationError(
            "factorized risk compilation supports backend='auto', 'classical', or 'pennylane'"
        )
    budget = StructuredRiskErrorBudget.allocate(target_error)
    if arithmetic_scale is not None and (not isfinite(arithmetic_scale) or arithmetic_scale <= 0):
        raise ValueError("arithmetic_scale must be finite and positive")
    exact_summary = evaluate_factor_risk(
        problem,
        chunk_size=validation_chunk_size,
        max_points=max_validation_points,
    )
    scales = (
        (arithmetic_scale,)
        if arithmetic_scale is not None
        else tuple(float(2**power) for power in range(0, 13))
    )
    selected_plan: StructuredLossOraclePlan | None = None
    selected_validation: StructuredRiskOracleValidation | None = None
    last_resource_error: ResourceLimitError | None = None
    for scale in scales:
        try:
            plan = compile_structured_loss_oracle(
                problem.model.encoding,
                problem.model.objective,
                loss_scale=scale,
                max_loss_qubits=max_loss_qubits,
                max_affine_output_qubits=max_affine_output_qubits,
            )
        except ResourceLimitError as exc:
            last_resource_error = exc
            continue
        validation = validate_structured_risk_oracle(
            problem,
            plan,
            exact_summary=exact_summary,
            chunk_size=validation_chunk_size,
            max_points=max_validation_points,
        )
        selected_plan = plan
        selected_validation = validation
        selected_error = (
            validation.value_at_risk_error
            if isinstance(problem, FactorVaR)
            else validation.expected_shortfall_error
        )
        if selected_error <= budget.loss_quantization:
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

    excess_qubits = selected_plan.loss_qubits if isinstance(problem, FactorCVaR) else 0
    ancillary_qubits = selected_plan.arithmetic_qubits + excess_qubits + 2
    strategy = compare_state_preparation_strategies(
        problem.model.encoding,
        target=representation_target,
        ancilla_qubits=ancillary_qubits,
        max_parameters=max_state_preparation_parameters,
        max_memory_bytes=max_state_preparation_memory_bytes,
    )
    total_wires = problem.model.encoding.total_qubits + ancillary_qubits
    selected_error = (
        selected_validation.value_at_risk_error
        if isinstance(problem, FactorVaR)
        else selected_validation.expected_shortfall_error
    )
    oracle_converged = selected_error <= budget.loss_quantization
    within_runtime_width = total_wires <= max_total_wires
    if backend == "pennylane":
        strategy.require_selected()
        if not within_runtime_width:
            raise ResourceLimitError(
                f"factorized risk circuit requires {total_wires} wires, "
                f"above max_total_wires={max_total_wires}"
            )
        if not oracle_converged:
            raise ResourceLimitError(
                "fixed-point loss register exceeds its allocated financial error"
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
    return CompiledFactorRiskModel(
        problem=problem,
        oracle=selected_plan,
        validation=selected_validation,
        classical_summary=exact_summary,
        target_error=target_error,
        error_budget=budget,
        state_preparation_strategy=strategy,
        validation_chunk_size=validation_chunk_size,
        backend_name=resolved_backend,
    )


__all__ = [
    "CompiledFactorRiskModel",
    "FactorQuantumObjectiveEstimate",
    "FactorQuantumRiskResult",
    "FactorQuantumVaRSearch",
    "StructuredRiskErrorBudget",
    "compile_factor_risk_problem",
]
