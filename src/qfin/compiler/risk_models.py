"""Compiled hybrid quantum-risk models and result objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from math import isfinite

import numpy as np

from qfin.algorithms import AmplitudeEstimate, maximum_likelihood_amplitude_estimate
from qfin.backends.risk import RiskPennyLaneBackend
from qfin.finance.fixed_income import Engine
from qfin.finance.risk import (
    CVaR,
    RiskConfidenceInterval,
    RiskSummary,
    TailProbability,
    TailProbabilitySummary,
    VaR,
    aggregate_risk,
    bootstrap_risk_interval,
    evaluate_tail_probability,
)
from qfin.representation import (
    DistributionEncoding,
    QuantumObjectiveEncoding,
    cdf_objective,
    tail_excess_objective,
    tail_probability_objective,
)
from qfin.resources import RiskProblemKind, RiskResourceReport, estimate_risk_resources

RiskProblem = TailProbability | VaR | CVaR


def _resolve_quantum_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    return "lightning.qubit" if find_spec("pennylane_lightning") is not None else "default.qubit"


@dataclass(frozen=True, slots=True)
class RiskErrorBudget:
    """Financial/probability error allocation for a compiled risk objective."""

    total: float
    distribution: float
    oracle: float
    estimation: float
    interval_level: float = 0.95

    @classmethod
    def allocate(cls, target_error: float) -> RiskErrorBudget:
        if not isfinite(target_error) or target_error <= 0:
            raise ValueError("target_error must be finite and greater than zero")
        return cls(
            total=target_error,
            distribution=0.5 * target_error,
            oracle=0.0,
            estimation=0.5 * target_error,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "total": self.total,
            "distribution": self.distribution,
            "oracle": self.oracle,
            "estimation": self.estimation,
            "interval_level": self.interval_level,
            "oracle_note": "exact grid-point indicator/excess rotations",
        }


@dataclass(frozen=True, slots=True)
class QuantumThresholdEstimate:
    """One MLAE CDF, tail, or excess objective evaluation."""

    objective: str
    threshold: float
    encoded_amplitude: float
    estimate: AmplitudeEstimate

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "threshold": self.threshold,
            "encoded_amplitude": self.encoded_amplitude,
            "estimate": self.estimate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class QuantumVaRSearch:
    """Hybrid binary search driven by quantum CDF-amplitude estimates."""

    confidence: float
    value: float
    confidence_interval_95: tuple[float, float]
    encoded_value: float
    evaluations: tuple[QuantumThresholdEstimate, ...]
    selected_estimate: QuantumThresholdEstimate
    resolved_to_single_grid_point: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "value": self.value,
            "confidence_interval_95": list(self.confidence_interval_95),
            "encoded_value": self.encoded_value,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "selected_estimate": self.selected_estimate.to_dict(),
            "resolved_to_single_grid_point": self.resolved_to_single_grid_point,
            "caveat": (
                "The interval combines local MLAE intervals through monotonicity; "
                "it is not a simultaneous-coverage guarantee."
            ),
        }


@dataclass(frozen=True, slots=True)
class QuantumRiskResult:
    """Quantum estimate with classical, encoded, error, and resource references."""

    problem_kind: RiskProblemKind
    value: float
    confidence_interval_95: tuple[float, float]
    classical_value: float
    encoded_value: float
    absolute_error: float
    representation_error: float
    estimation_error: float
    target_error: float
    meets_target_error: bool
    threshold: float | None
    tail_probability: float | None
    tail_probability_interval_95: tuple[float, float] | None
    value_at_risk: float | None
    value_at_risk_interval_95: tuple[float, float] | None
    expected_shortfall: float | None
    amplitude_estimates: tuple[QuantumThresholdEstimate, ...]
    resources: RiskResourceReport
    classical_interval: RiskConfidenceInterval | None
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
            "representation_error": self.representation_error,
            "estimation_error": self.estimation_error,
            "target_error": self.target_error,
            "meets_target_error": self.meets_target_error,
            "threshold": self.threshold,
            "tail_probability": self.tail_probability,
            "tail_probability_interval_95": (
                None
                if self.tail_probability_interval_95 is None
                else list(self.tail_probability_interval_95)
            ),
            "value_at_risk": self.value_at_risk,
            "value_at_risk_interval_95": (
                None
                if self.value_at_risk_interval_95 is None
                else list(self.value_at_risk_interval_95)
            ),
            "expected_shortfall": self.expected_shortfall,
            "amplitude_estimates": [item.to_dict() for item in self.amplitude_estimates],
            "resources": self.resources.to_dict(),
            "classical_interval": (
                None if self.classical_interval is None else self.classical_interval.to_dict()
            ),
            "backend": self.backend,
            "algorithm": self.algorithm,
            "caveat": (
                "Experimental simulator workflow with generic O(2**n) state and "
                "objective loading; no quantum-advantage claim. CVaR intervals are "
                "conditional on the selected VaR grid point, and VaR intervals combine "
                "local rather than simultaneous MLAE coverage."
            ),
        }


@dataclass(frozen=True, slots=True)
class CompiledRiskModel:
    """Compiled tail-probability, VaR, or CVaR financial problem."""

    problem: RiskProblem
    representation: DistributionEncoding
    target_error: float
    error_budget: RiskErrorBudget
    problem_kind: RiskProblemKind
    classical_value: float
    encoded_value: float
    representation_error: float
    representation_converged: bool
    backend_name: str = "pennylane"
    algorithm_name: str = "maximum_likelihood_amplitude_estimation"
    quantum_algorithm_available: bool = True

    @property
    def compilation_converged(self) -> bool:
        return self.representation_converged

    def run(self, *, engine: Engine = "auto") -> RiskSummary | TailProbabilitySummary:
        """Run the stable classical calculation retained for validation."""

        if isinstance(self.problem, TailProbability):
            if engine not in ("auto", "numpy"):
                raise ValueError("tail probability classical execution uses NumPy")
            return evaluate_tail_probability(self.problem)
        return aggregate_risk(
            self.problem.distribution,
            confidence=self.problem.confidence,
            engine=engine,
        )

    def to_pennylane(
        self,
        *,
        device_name: str = "auto",
        max_structured_rotations: int = 32_767,
    ) -> RiskPennyLaneBackend:
        """Build the PennyLane risk runtime without duplicating its simulator."""

        if self.backend_name != "pennylane":
            raise ValueError(f"compiled backend is {self.backend_name!r}, not 'pennylane'")
        return RiskPennyLaneBackend(
            self.representation,
            device_name=_resolve_quantum_device(device_name),
            max_structured_rotations=max_structured_rotations,
        )

    def resources(
        self,
        *,
        schedule: Sequence[int] = (0, 1, 2, 4),
        shots: int = 1_000,
        device_name: str = "auto",
        threshold_evaluations: int | None = None,
    ) -> RiskResourceReport:
        probabilities = self.representation.probabilities
        occupied = int(np.count_nonzero(probabilities > 0))
        resolved_device = _resolve_quantum_device(device_name)
        return estimate_risk_resources(
            self.representation.qubits,
            input_points=self.problem.distribution.losses.size,
            occupied_grid_points=occupied,
            problem_kind=self.problem_kind,
            schedule=schedule,
            shots=shots,
            backend=f"pennylane.{resolved_device}",
            threshold_evaluations=threshold_evaluations,
        )

    @staticmethod
    def _estimate_objective(
        runtime: RiskPennyLaneBackend,
        objective: QuantumObjectiveEncoding,
        *,
        schedule: tuple[int, ...],
        shots: int,
        seed: int | None,
        likelihood_grid_size: int,
    ) -> QuantumThresholdEstimate:
        observations = runtime.run_schedule(
            objective,
            schedule,
            shots=shots,
            seed=seed,
        )
        estimate = maximum_likelihood_amplitude_estimate(
            observations,
            grid_size=likelihood_grid_size,
        )
        assert objective.threshold is not None
        return QuantumThresholdEstimate(
            objective=objective.label,
            threshold=objective.threshold,
            encoded_amplitude=objective.exact_amplitude,
            estimate=estimate,
        )

    def _run_var_search(
        self,
        runtime: RiskPennyLaneBackend,
        *,
        confidence: float,
        schedule: tuple[int, ...],
        shots: int,
        seed: int | None,
        likelihood_grid_size: int,
    ) -> QuantumVaRSearch:
        occupied = self.representation.probabilities > 0
        candidates = np.unique(self.representation.grid[occupied])
        if candidates.size == 0:
            raise ValueError("risk representation contains no occupied grid points")
        lower_index = 0
        upper_index = candidates.size - 1
        evaluations: list[QuantumThresholdEstimate] = []
        evaluated_indices: dict[int, QuantumThresholdEstimate] = {}
        while lower_index < upper_index:
            middle = (lower_index + upper_index) // 2
            objective = cdf_objective(self.representation, float(candidates[middle]))
            estimate = self._estimate_objective(
                runtime,
                objective,
                schedule=schedule,
                shots=shots,
                seed=None if seed is None else seed + len(evaluations),
                likelihood_grid_size=likelihood_grid_size,
            )
            evaluations.append(estimate)
            evaluated_indices[middle] = estimate
            if estimate.estimate.amplitude >= confidence:
                upper_index = middle
            else:
                lower_index = middle + 1

        selected_index = lower_index
        if selected_index not in evaluated_indices:
            objective = cdf_objective(self.representation, float(candidates[selected_index]))
            final_estimate = self._estimate_objective(
                runtime,
                objective,
                schedule=schedule,
                shots=shots,
                seed=None if seed is None else seed + len(evaluations),
                likelihood_grid_size=likelihood_grid_size,
            )
            evaluations.append(final_estimate)
            evaluated_indices[selected_index] = final_estimate

        interval_lower = 0
        interval_upper = candidates.size - 1
        for index, item in evaluated_indices.items():
            if item.estimate.upper_95 < confidence:
                interval_lower = max(interval_lower, min(index + 1, candidates.size - 1))
            if item.estimate.lower_95 >= confidence:
                interval_upper = min(interval_upper, index)
        if interval_lower > interval_upper:
            interval_lower = selected_index
            interval_upper = selected_index
        return QuantumVaRSearch(
            confidence=confidence,
            value=float(candidates[selected_index]),
            confidence_interval_95=(
                float(candidates[interval_lower]),
                float(candidates[interval_upper]),
            ),
            encoded_value=self.encoded_value
            if isinstance(self.problem, VaR)
            else float(_encoded_risk_value(self.representation, VaR, confidence)),
            evaluations=tuple(evaluations),
            selected_estimate=evaluated_indices[selected_index],
            resolved_to_single_grid_point=interval_lower == interval_upper,
        )

    def run_quantum(
        self,
        *,
        shots: int = 2_000,
        schedule: Sequence[int] = (0, 1, 2, 4),
        seed: int | None = None,
        likelihood_grid_size: int = 131_073,
        max_structured_rotations: int = 32_767,
        device_name: str = "auto",
        bootstrap_resamples: int = 0,
        bootstrap_seed: int | None = 0,
    ) -> QuantumRiskResult:
        """Execute the experimental MLAE tail-risk workflow."""

        if bootstrap_resamples == 1 or bootstrap_resamples < 0:
            raise ValueError("bootstrap_resamples must be zero or at least two")
        powers = tuple(int(power) for power in schedule)
        resolved_device = _resolve_quantum_device(device_name)
        runtime = self.to_pennylane(
            device_name=resolved_device,
            max_structured_rotations=max_structured_rotations,
        )
        classical_interval: RiskConfidenceInterval | None = None
        if bootstrap_resamples:
            if isinstance(self.problem, TailProbability):
                raise ValueError("bootstrap intervals are implemented for VaR/CVaR only")
            classical_interval = bootstrap_risk_interval(
                self.problem.distribution,
                confidence=self.problem.confidence,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )

        if isinstance(self.problem, TailProbability):
            objective = tail_probability_objective(
                self.representation,
                self.problem.threshold,
                inclusive=self.problem.inclusive,
            )
            estimate_item = self._estimate_objective(
                runtime,
                objective,
                schedule=powers,
                shots=shots,
                seed=seed,
                likelihood_grid_size=likelihood_grid_size,
            )
            estimate = estimate_item.estimate
            value = estimate.amplitude
            interval = (estimate.lower_95, estimate.upper_95)
            resources = self.resources(
                schedule=powers,
                shots=shots,
                device_name=resolved_device,
                threshold_evaluations=1,
            )
            return self._build_result(
                value=value,
                interval=interval,
                threshold=self.problem.threshold,
                tail_probability=value,
                tail_interval=interval,
                value_at_risk=None,
                var_interval=None,
                expected_shortfall=None,
                estimates=(estimate_item,),
                resources=resources,
                classical_interval=None,
                backend=resolved_device,
            )

        search = self._run_var_search(
            runtime,
            confidence=self.problem.confidence,
            schedule=powers,
            shots=shots,
            seed=seed,
            likelihood_grid_size=likelihood_grid_size,
        )
        final_cdf = search.selected_estimate.estimate
        tail_probability = 1.0 - final_cdf.amplitude
        tail_interval = (1.0 - final_cdf.upper_95, 1.0 - final_cdf.lower_95)

        if isinstance(self.problem, VaR):
            resources = self.resources(
                schedule=powers,
                shots=shots,
                device_name=resolved_device,
                threshold_evaluations=len(search.evaluations),
            )
            return self._build_result(
                value=search.value,
                interval=search.confidence_interval_95,
                threshold=search.value,
                tail_probability=tail_probability,
                tail_interval=tail_interval,
                value_at_risk=search.value,
                var_interval=search.confidence_interval_95,
                expected_shortfall=None,
                estimates=search.evaluations,
                resources=resources,
                classical_interval=classical_interval,
                backend=resolved_device,
            )

        excess_objective = tail_excess_objective(self.representation, search.value)
        excess_item = self._estimate_objective(
            runtime,
            excess_objective,
            schedule=powers,
            shots=shots,
            seed=None if seed is None else seed + len(search.evaluations) + 10_000,
            likelihood_grid_size=likelihood_grid_size,
        )
        multiplier = excess_objective.financial_scale / (1.0 - self.problem.confidence)
        value = search.value + multiplier * excess_item.estimate.amplitude
        conditional_interval = (
            search.value + multiplier * excess_item.estimate.lower_95,
            search.value + multiplier * excess_item.estimate.upper_95,
        )
        estimates = (*search.evaluations, excess_item)
        resources = self.resources(
            schedule=powers,
            shots=shots,
            device_name=resolved_device,
            threshold_evaluations=len(search.evaluations),
        )
        return self._build_result(
            value=value,
            interval=conditional_interval,
            threshold=search.value,
            tail_probability=tail_probability,
            tail_interval=tail_interval,
            value_at_risk=search.value,
            var_interval=search.confidence_interval_95,
            expected_shortfall=value,
            estimates=estimates,
            resources=resources,
            classical_interval=classical_interval,
            backend=resolved_device,
        )

    def _build_result(
        self,
        *,
        value: float,
        interval: tuple[float, float],
        threshold: float | None,
        tail_probability: float | None,
        tail_interval: tuple[float, float] | None,
        value_at_risk: float | None,
        var_interval: tuple[float, float] | None,
        expected_shortfall: float | None,
        estimates: tuple[QuantumThresholdEstimate, ...],
        resources: RiskResourceReport,
        classical_interval: RiskConfidenceInterval | None,
        backend: str,
    ) -> QuantumRiskResult:
        absolute_error = abs(value - self.classical_value)
        estimation_error = abs(value - self.encoded_value)
        return QuantumRiskResult(
            problem_kind=self.problem_kind,
            value=value,
            confidence_interval_95=(min(interval), max(interval)),
            classical_value=self.classical_value,
            encoded_value=self.encoded_value,
            absolute_error=absolute_error,
            representation_error=self.representation_error,
            estimation_error=estimation_error,
            target_error=self.target_error,
            meets_target_error=absolute_error <= self.target_error,
            threshold=threshold,
            tail_probability=tail_probability,
            tail_probability_interval_95=(
                None if tail_interval is None else (min(tail_interval), max(tail_interval))
            ),
            value_at_risk=value_at_risk,
            value_at_risk_interval_95=(
                None if var_interval is None else (min(var_interval), max(var_interval))
            ),
            expected_shortfall=expected_shortfall,
            amplitude_estimates=estimates,
            resources=resources,
            classical_interval=classical_interval,
            backend=f"pennylane.{backend}:structured",
            algorithm=self.algorithm_name,
        )

    def explain(self) -> str:
        convergence = "met" if self.representation_converged else "not met at max_qubits"
        if self.problem_kind == "tail_probability":
            algorithm = "one threshold-indicator MLAE objective"
        elif self.problem_kind == "value_at_risk":
            algorithm = "hybrid binary search over MLAE CDF objectives"
        else:
            algorithm = "MLAE CDF search plus an MLAE normalized tail-excess objective"
        return (
            f"QFin compiled {self.problem_kind} over an empirical loss distribution.\n"
            f"Representation: {self.representation.qubits} data qubits, "
            f"{self.representation.grid_points} grid points; error="
            f"{self.representation_error:.6g} ({convergence}).\n"
            f"Algorithm: {algorithm}.\n"
            "State preparation: generic probability-tree multiplexed RY loading "
            "with O(2**data_qubits) parameters.\n"
            "Execution: PennyLane with Lightning preferred; this is an experimental "
            "simulator workflow and does not claim quantum advantage."
        )


def _encoded_risk_value(
    representation: DistributionEncoding,
    problem_type: type[VaR] | type[CVaR],
    confidence: float,
) -> float:
    """Return the encoded finite-distribution VaR or CVaR reference."""

    from qfin.finance.risk import LossDistribution

    summary = aggregate_risk(
        LossDistribution(representation.grid, representation.probabilities),
        confidence=confidence,
        engine="numpy",
    )
    return summary.var if problem_type is VaR else summary.cvar


__all__ = [
    "CompiledRiskModel",
    "QuantumRiskResult",
    "QuantumThresholdEstimate",
    "QuantumVaRSearch",
    "RiskErrorBudget",
    "RiskProblem",
]
