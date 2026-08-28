"""Compile ALM scenario risk measures into PennyLane amplitude estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite, log2
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.algorithms import AmplitudeEstimate, maximum_likelihood_amplitude_estimate
from qfin.backends import (
    CompressedPennyLaneBackend,
    DensePennyLaneBackend,
    StructuredPennyLaneBackend,
)
from qfin.backends.runtime import ComplexPrecision
from qfin.circuits import WalshPayoffApproximation
from qfin.compiler.models import ErrorBudget
from qfin.exceptions import CompilationError
from qfin.finance.alm import ALMScenarioResult, AssetLiabilityModel
from qfin.representation import DistributionEncoding
from qfin.resources import BackendMode, ResourceReport, estimate_resources

ALMRiskMetric = Literal["shortfall_probability", "expected_shortfall"]
PennyLaneALMRuntime = (
    CompressedPennyLaneBackend | StructuredPennyLaneBackend | DensePennyLaneBackend
)


@dataclass(frozen=True, slots=True)
class ALMRiskResult:
    """Quantum ALM estimate with exact scenario and resource benchmarks."""

    value: float
    confidence_interval_95: tuple[float, float]
    classical_value: float
    circuit_value: float
    absolute_error: float
    payoff_approximation_error: float
    estimation_error: float
    target_error: float
    meets_target_error: bool
    metric: ALMRiskMetric
    amplitude: AmplitudeEstimate
    resources: ResourceReport
    payoff_approximation: WalshPayoffApproximation | None
    backend: str
    algorithm: str

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "confidence_interval_95": list(self.confidence_interval_95),
            "classical_value": self.classical_value,
            "circuit_value": self.circuit_value,
            "absolute_error": self.absolute_error,
            "payoff_approximation_error": self.payoff_approximation_error,
            "estimation_error": self.estimation_error,
            "target_error": self.target_error,
            "meets_target_error": self.meets_target_error,
            "metric": self.metric,
            "amplitude_estimation": self.amplitude.to_dict(),
            "resources": self.resources.to_dict(),
            "payoff_approximation": (
                None
                if self.payoff_approximation is None
                else self.payoff_approximation.to_dict()
            ),
            "backend": self.backend,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True, slots=True)
class CompiledALMRiskModel:
    """Scenario-to-circuit compilation for one ALM shortfall measure."""

    alm_model: AssetLiabilityModel
    scenarios: ALMScenarioResult
    metric: ALMRiskMetric
    representation: DistributionEncoding
    raw_objective: NDArray[np.float64]
    normalized_objective: NDArray[np.float64]
    objective_scale: float
    classical_value: float
    target_error: float
    error_budget: ErrorBudget
    payoff_approximation: WalshPayoffApproximation | None
    circuit_value: float
    payoff_approximation_error: float
    backend_name: str = "pennylane"
    algorithm_name: str = "maximum_likelihood_amplitude_estimation"

    @property
    def compilation_converged(self) -> bool:
        return (
            self.payoff_approximation is None
            or self.payoff_approximation.met_tolerance
        )

    def _resolve_backend_mode(self, mode: BackendMode | None) -> BackendMode:
        if mode is None:
            return "compressed" if self.payoff_approximation is not None else "structured"
        return mode

    def resources(
        self,
        *,
        schedule: Sequence[int] = (0, 1, 2, 4),
        shots: int = 1_000,
        backend_mode: BackendMode | None = None,
        device_name: str = "lightning.qubit",
    ) -> ResourceReport:
        mode = self._resolve_backend_mode(backend_mode)
        return estimate_resources(
            self.representation.qubits,
            schedule=schedule,
            shots=shots,
            backend=f"pennylane.{device_name}",
            backend_mode=mode,
            payoff_terms=(
                self.payoff_approximation.parameter_count
                if mode == "compressed" and self.payoff_approximation is not None
                else None
            ),
        )

    def to_pennylane(
        self,
        *,
        mode: BackendMode | None = None,
        max_dense_dimension: int = 2_048,
        max_structured_rotations: int = 32_767,
        max_compressed_terms: int = 32_767,
        device_name: str = "lightning.qubit",
        precision: ComplexPrecision = "complex128",
    ) -> PennyLaneALMRuntime:
        if self.backend_name != "pennylane":
            raise ValueError(f"compiled backend is {self.backend_name!r}, not 'pennylane'")
        resolved = self._resolve_backend_mode(mode)
        if resolved == "compressed":
            if self.payoff_approximation is None:
                raise ValueError("compressed ALM execution requires uniform scenarios")
            return CompressedPennyLaneBackend(
                self.representation,
                self.normalized_objective,
                self.payoff_approximation,
                device_name=device_name,
                precision=precision,
                max_compressed_terms=max_compressed_terms,
            )
        if resolved == "structured":
            return StructuredPennyLaneBackend(
                self.representation,
                self.normalized_objective,
                device_name=device_name,
                precision=precision,
                max_structured_rotations=max_structured_rotations,
            )
        if resolved == "dense":
            return DensePennyLaneBackend(
                self.representation,
                self.normalized_objective,
                device_name=device_name,
                precision=precision,
                max_dense_dimension=max_dense_dimension,
            )
        raise ValueError("mode must be 'compressed', 'structured', or 'dense'")

    def explain(self) -> str:
        mode = self._resolve_backend_mode(None)
        approximation = (
            "exact probability-tree objective loading"
            if self.payoff_approximation is None
            else (
                f"{self.payoff_approximation.parameter_count}/"
                f"{self.payoff_approximation.full_term_count} Walsh terms; "
                f"financial error={self.payoff_approximation_error:.6g}"
            )
        )
        return (
            f"QFin compiled ALM metric={self.metric} across "
            f"{self.scenarios.parallel_shocks.size} rate scenarios.\n"
            f"Scenario register: {self.representation.qubits} data qubits; "
            f"state preparation={self.representation.state_preparation_method}.\n"
            f"Circuit backend={mode}; objective loading={approximation}.\n"
            f"Classical scenario value={self.classical_value:.6g}; "
            f"compiled-circuit value={self.circuit_value:.6g}.\n"
            "Execution uses PennyLane with Lightning by default; resource counts are "
            "logical simulator estimates, not evidence of hardware advantage."
        )

    def run(
        self,
        *,
        shots: int = 1_000,
        schedule: Sequence[int] = (0, 1, 2, 4),
        seed: int | None = None,
        likelihood_grid_size: int = 131_073,
        backend_mode: BackendMode | None = None,
        max_dense_dimension: int = 2_048,
        max_structured_rotations: int = 32_767,
        max_compressed_terms: int = 32_767,
        device_name: str = "lightning.qubit",
        precision: ComplexPrecision = "complex128",
    ) -> ALMRiskResult:
        powers = tuple(int(power) for power in schedule)
        mode = self._resolve_backend_mode(backend_mode)
        runtime = self.to_pennylane(
            mode=mode,
            max_dense_dimension=max_dense_dimension,
            max_structured_rotations=max_structured_rotations,
            max_compressed_terms=max_compressed_terms,
            device_name=device_name,
            precision=precision,
        )
        observations = runtime.run_schedule(powers, shots=shots, seed=seed)
        amplitude = maximum_likelihood_amplitude_estimate(
            observations, grid_size=likelihood_grid_size
        )
        value = self.objective_scale * amplitude.amplitude
        lower = self.objective_scale * amplitude.lower_95
        upper = self.objective_scale * amplitude.upper_95
        reference = self.circuit_value if mode == "compressed" else self.classical_value
        payoff_error = abs(reference - self.classical_value)
        absolute_error = abs(value - self.classical_value)
        resources = self.resources(
            schedule=powers,
            shots=shots,
            backend_mode=mode,
            device_name=runtime.resolved_device_name,
        )
        return ALMRiskResult(
            value=value,
            confidence_interval_95=(min(lower, upper), max(lower, upper)),
            classical_value=self.classical_value,
            circuit_value=reference,
            absolute_error=absolute_error,
            payoff_approximation_error=payoff_error,
            estimation_error=abs(value - reference),
            target_error=self.target_error,
            meets_target_error=absolute_error <= self.target_error,
            metric=self.metric,
            amplitude=amplitude,
            resources=resources,
            payoff_approximation=(
                self.payoff_approximation if mode == "compressed" else None
            ),
            backend=f"pennylane.{runtime.resolved_device_name}:{mode}",
            algorithm=self.algorithm_name,
        )


def compile_alm(
    model: AssetLiabilityModel,
    parallel_shocks: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
    metric: ALMRiskMetric = "expected_shortfall",
    target_error: float = 0.01,
    backend: str = "pennylane",
    max_qubits: int = 12,
    payoff_angle_tolerance: float = 0.1,
    payoff_max_terms: int | None = None,
    max_working_bytes: int = 64 * 1024 * 1024,
) -> CompiledALMRiskModel:
    """Compile an ALM shortfall expectation over discrete rate scenarios."""
    if not isinstance(model, AssetLiabilityModel):
        raise CompilationError("compile_alm requires an AssetLiabilityModel")
    if metric not in ("shortfall_probability", "expected_shortfall"):
        raise ValueError(
            "metric must be 'shortfall_probability' or 'expected_shortfall'"
        )
    if backend != "pennylane":
        raise CompilationError("QFin ALM currently supports backend='pennylane' only")
    if not isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be finite and positive")
    if isinstance(max_qubits, bool) or max_qubits < 1:
        raise ValueError("max_qubits must be positive")
    if not isfinite(payoff_angle_tolerance) or payoff_angle_tolerance <= 0:
        raise ValueError("payoff_angle_tolerance must be finite and positive")
    if payoff_max_terms is not None and payoff_max_terms < 1:
        raise ValueError("payoff_max_terms must be positive")

    scenarios = model.run_parallel_shocks(
        parallel_shocks,
        probabilities=probabilities,
        max_working_bytes=max_working_bytes,
    )
    count = int(scenarios.surplus.size)
    qubits = max(1, ceil(log2(count)))
    if qubits > max_qubits:
        raise CompilationError(
            f"{count} scenarios require {qubits} qubits, above max_qubits={max_qubits}"
        )
    points = 2**qubits
    order = np.argsort(scenarios.surplus, kind="stable")
    sorted_surplus = scenarios.surplus[order]
    sorted_probabilities = scenarios.probabilities[order]
    objective = (
        np.asarray(sorted_surplus < 0, dtype=np.float64)
        if metric == "shortfall_probability"
        else np.maximum(-sorted_surplus, 0.0)
    )
    scale = 1.0 if metric == "shortfall_probability" else float(np.max(objective))
    if scale == 0.0:
        scale = 1.0

    grid = np.zeros(points, dtype=np.float64)
    encoded_probabilities = np.zeros(points, dtype=np.float64)
    raw_objective = np.zeros(points, dtype=np.float64)
    grid[:count] = sorted_surplus
    encoded_probabilities[:count] = sorted_probabilities
    raw_objective[:count] = objective
    normalized = raw_objective / scale
    uniform = count == points and np.allclose(
        sorted_probabilities, 1.0 / count, atol=1e-14
    )
    representation = DistributionEncoding(
        grid=grid,
        probabilities=encoded_probabilities,
        qubits=qubits,
        lower_bound=float(np.min(sorted_surplus)),
        upper_bound=float(np.max(sorted_surplus)),
        tail_probability=0.0,
        discretization_error=0.0,
        mean_error=0.0,
        objective=metric,
        encoding_method=("uniform_scenario_index" if uniform else "scenario_probability"),
        state_preparation_method=(
            "uniform_scenario_hadamard"
            if uniform
            else "probability_tree_multiplexed_ry"
        ),
    )
    classical_value = float(np.dot(encoded_probabilities, raw_objective))
    budget = ErrorBudget.allocate(target_error)
    approximation: WalshPayoffApproximation | None = None
    if uniform:
        approximation = WalshPayoffApproximation.fit(
            normalized,
            financial_multiplier=scale,
            target_price_error=budget.algorithmic,
            max_angle_rmse=payoff_angle_tolerance,
            max_terms=(
                None if payoff_max_terms is None else min(payoff_max_terms, points)
            ),
        )
    circuit_value = (
        classical_value
        if approximation is None
        else scale * approximation.approximate_amplitude
    )
    raw_objective.setflags(write=False)
    normalized.setflags(write=False)
    return CompiledALMRiskModel(
        alm_model=model,
        scenarios=scenarios,
        metric=metric,
        representation=representation,
        raw_objective=raw_objective,
        normalized_objective=normalized,
        objective_scale=scale,
        classical_value=classical_value,
        target_error=target_error,
        error_budget=budget,
        payoff_approximation=approximation,
        circuit_value=circuit_value,
        payoff_approximation_error=abs(circuit_value - classical_value),
        backend_name=backend,
    )
