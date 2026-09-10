"""Compiled model and result objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import AmplitudeEstimate, maximum_likelihood_amplitude_estimate
from qfin.algorithms.amplitude_estimation import _validated_schedule
from qfin.backends import (
    CompressedPennyLaneBackend,
    DensePennyLaneBackend,
    StructuredPennyLaneBackend,
)
from qfin.backends.devices import DeviceTarget, resolve_quantum_device
from qfin.backends.interop import QasmExport, export_openqasm, export_qiskit
from qfin.backends.noise import NoiseMitigationReport, NoiseModel, analyze_noise
from qfin.circuits import WalshPayoffApproximation
from qfin.compiler.risk_models import CompiledRiskModel as CompiledRiskModel
from qfin.finance import BlackScholes, EuropeanOption, LogNormal
from qfin.representation import DistributionEncoding
from qfin.representation.strategies import StatePreparationStrategyReport
from qfin.resources import (
    BackendMode,
    DeviceResourceReport,
    ResourceReport,
    estimate_device_resources,
    estimate_resources,
)

PennyLaneRuntime = CompressedPennyLaneBackend | StructuredPennyLaneBackend | DensePennyLaneBackend


@dataclass(frozen=True, slots=True)
class ErrorBudget:
    """Requested financial-unit tolerance split across compiler stages."""

    total: float
    domain_truncation: float
    discretization: float
    algorithmic: float
    sampling: float
    target_error_unit: str = "currency / price units"

    def __post_init__(self) -> None:
        components = (
            self.domain_truncation,
            self.discretization,
            self.algorithmic,
            self.sampling,
        )
        if not isfinite(self.total) or self.total <= 0.0:
            raise ValueError("target_error must be finite and greater than zero")
        if any(not isfinite(value) or value < 0.0 for value in components):
            raise ValueError("error-budget allocations must be finite and non-negative")
        if not np.isclose(sum(components), self.total, rtol=1.0e-12, atol=0.0):
            raise ValueError("error-budget allocations must sum to total")
        if not self.target_error_unit.strip():
            raise ValueError("target_error_unit must not be empty")

    @classmethod
    def allocate(
        cls,
        target_error: float,
        *,
        target_error_unit: str = "currency / price units",
    ) -> ErrorBudget:
        if not isfinite(target_error) or target_error <= 0.0:
            raise ValueError("target_error must be finite and greater than zero")
        return cls(
            total=target_error,
            domain_truncation=0.15 * target_error,
            discretization=0.35 * target_error,
            algorithmic=0.25 * target_error,
            sampling=0.25 * target_error,
            target_error_unit=target_error_unit,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "total": self.total,
            "domain_truncation": self.domain_truncation,
            "discretization": self.discretization,
            "algorithmic": self.algorithmic,
            "sampling": self.sampling,
            "target_error_unit": self.target_error_unit,
        }


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Financial result plus quantum, validation, error, and resource metadata."""

    value: float
    confidence_interval_95: tuple[float, float]
    classical_value: float
    discrete_value: float
    circuit_value: float
    absolute_error: float
    representation_error: float
    payoff_approximation_error: float
    estimation_error: float
    target_error: float
    target_error_unit: str
    meets_target_error: bool
    amplitude: AmplitudeEstimate
    resources: ResourceReport
    payoff_approximation: WalshPayoffApproximation | None
    backend: str
    algorithm: str

    def to_dict(self) -> dict[str, object]:
        return {
            "problem_category": "option_pricing",
            "financial_objective": "european_option_price",
            "representation": self.resources.backend_mode,
            "algorithm_error": self.payoff_approximation_error,
            "sampling_statistical_error": max(
                abs(self.value - self.confidence_interval_95[0]),
                abs(self.confidence_interval_95[1] - self.value),
            ),
            "sampling_error_definition": "maximum distance to the reported 95% interval endpoints",
            "quantum_execution_available": True,
            "resource_estimate_type": "logical_circuit_counts",
            "value": self.value,
            "confidence_interval_95": list(self.confidence_interval_95),
            "classical_value": self.classical_value,
            "discrete_value": self.discrete_value,
            "circuit_value": self.circuit_value,
            "absolute_error": self.absolute_error,
            "representation_error": self.representation_error,
            "payoff_approximation_error": self.payoff_approximation_error,
            "estimation_error": self.estimation_error,
            "target_error": self.target_error,
            "target_error_unit": self.target_error_unit,
            "meets_target_error": self.meets_target_error,
            "amplitude_estimation": self.amplitude.to_dict(),
            "resources": self.resources.to_dict(),
            "payoff_approximation": (
                None if self.payoff_approximation is None else self.payoff_approximation.to_dict()
            ),
            "backend": self.backend,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True, slots=True)
class CompiledPricingModel:
    """Complete finance-to-quantum compilation for one European option."""

    instrument: EuropeanOption
    market: BlackScholes
    distribution: LogNormal
    representation: DistributionEncoding
    raw_payoff: NDArray[np.float64]
    normalized_payoff: NDArray[np.float64]
    payoff_scale: float
    discount_factor: float
    discrete_value: float
    classical_value: float
    target_error: float
    error_budget: ErrorBudget
    representation_error: float
    representation_converged: bool
    payoff_approximation: WalshPayoffApproximation | None
    circuit_value: float
    payoff_approximation_error: float
    representation_method: str
    state_preparation_strategy: StatePreparationStrategyReport
    backend_name: str = "pennylane"
    algorithm_name: str = "maximum_likelihood_amplitude_estimation"

    def to_dict(self) -> dict[str, object]:
        """Describe compilation without executing a circuit or inventing shot error."""

        return {
            "problem_category": "option_pricing",
            "financial_objective": f"european_{self.instrument.kind}_price",
            "backend": self.backend_name,
            "representation": self.representation_method,
            "algorithm": self.algorithm_name,
            "target_error": self.target_error,
            "target_error_unit": self.target_error_unit,
            "representation_error": self.representation_error,
            "discretization_error": self.representation.discretization_error,
            "algorithm_error": self.payoff_approximation_error,
            "sampling_statistical_error": None,
            "compilation_converged": self.compilation_converged,
            "quantum_execution_available": self.backend_name == "pennylane",
            "resource_estimate_type": "logical_circuit_counts",
            "error_budget": self.error_budget.to_dict(),
            "limitations": [
                "Experimental simulator workflow; optional quantum dependencies required.",
                "Logical counts are not hardware runtime or evidence of quantum advantage.",
                "Representation error is checked against the Black-Scholes reference.",
            ],
        }

    @property
    def target_error_unit(self) -> str:
        """Unit attached to every stage of this compilation budget."""

        return self.error_budget.target_error_unit

    @property
    def compilation_converged(self) -> bool:
        payoff_converged = (
            self.payoff_approximation is None or self.payoff_approximation.met_tolerance
        )
        return self.representation_converged and payoff_converged

    def resources(
        self,
        *,
        schedule: Sequence[int] = (0, 1, 2, 4),
        shots: int = 1_000,
        backend_mode: BackendMode | None = None,
        device_name: str = "auto",
    ) -> ResourceReport:
        resolved_mode = self._resolve_backend_mode(backend_mode)
        resolved_device = resolve_quantum_device(device_name, require_available=False)
        return estimate_resources(
            self.representation.qubits,
            schedule=schedule,
            shots=shots,
            backend=f"pennylane.{resolved_device}",
            backend_mode=resolved_mode,
            payoff_terms=(
                self.payoff_approximation.parameter_count
                if resolved_mode == "compressed" and self.payoff_approximation is not None
                else None
            ),
        )

    def _resolve_backend_mode(self, mode: BackendMode | None) -> BackendMode:
        if mode is None:
            return "compressed" if self.payoff_approximation is not None else "structured"
        return mode

    def to_pennylane(
        self,
        *,
        mode: BackendMode | None = None,
        max_dense_dimension: int = 2_048,
        max_structured_rotations: int = 32_767,
        max_compressed_terms: int = 32_767,
        device_name: str = "auto",
    ) -> PennyLaneRuntime:
        """Build the optional PennyLane runtime adapter."""
        if self.backend_name != "pennylane":
            raise ValueError(f"compiled backend is {self.backend_name!r}, not 'pennylane'")
        resolved_mode = self._resolve_backend_mode(mode)
        resolved_device = resolve_quantum_device(device_name)
        if resolved_mode == "compressed":
            if self.payoff_approximation is None:
                raise ValueError("compressed backend requires representation_method='quantile'")
            return CompressedPennyLaneBackend(
                self.representation,
                self.normalized_payoff,
                self.payoff_approximation,
                device_name=resolved_device,
                max_compressed_terms=max_compressed_terms,
            )
        if resolved_mode == "structured":
            return StructuredPennyLaneBackend(
                self.representation,
                self.normalized_payoff,
                device_name=resolved_device,
                max_structured_rotations=max_structured_rotations,
            )
        if resolved_mode == "dense":
            return DensePennyLaneBackend(
                self.representation,
                self.normalized_payoff,
                device_name=resolved_device,
                max_dense_dimension=max_dense_dimension,
            )
        raise ValueError("mode must be 'compressed', 'structured', or 'dense'")

    def _portable_runtime(
        self,
        *,
        mode: BackendMode | None = None,
        max_structured_rotations: int = 32_767,
        max_compressed_terms: int = 32_767,
    ) -> CompressedPennyLaneBackend | StructuredPennyLaneBackend:
        resolved_mode = self._resolve_backend_mode(mode)
        if resolved_mode == "dense":
            raise ValueError(
                "dense QubitUnitary is a numerical reference and cannot be used for "
                "portable device analysis or export"
            )
        runtime = self.to_pennylane(
            mode=resolved_mode,
            max_structured_rotations=max_structured_rotations,
            max_compressed_terms=max_compressed_terms,
        )
        if not isinstance(runtime, (CompressedPennyLaneBackend, StructuredPennyLaneBackend)):
            raise TypeError("portable runtime unexpectedly resolved to the dense backend")
        return runtime

    def device_resources(
        self,
        *,
        schedule: Sequence[int] = (0, 1, 2, 4),
        shots: int = 1_000,
        target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
        backend_mode: BackendMode | None = None,
        max_structured_rotations: int = 32_767,
        max_compressed_terms: int = 32_767,
    ) -> DeviceResourceReport:
        """Return gate-set and routing resources for a portable target."""

        runtime = self._portable_runtime(
            mode=backend_mode,
            max_structured_rotations=max_structured_rotations,
            max_compressed_terms=max_compressed_terms,
        )
        return estimate_device_resources(
            runtime,
            schedule=schedule,
            shots=shots,
            target=target,
        )

    def noise_analysis(
        self,
        noise_model: NoiseModel,
        *,
        power: int = 0,
        shots: int | None = None,
        seed: int | None = 0,
        scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
        extrapolation_order: int = 1,
        backend_mode: BackendMode | None = None,
    ) -> NoiseMitigationReport:
        """Evaluate an explicit ``default.mixed`` model and polynomial ZNE."""

        runtime = self._portable_runtime(mode=backend_mode)
        return analyze_noise(
            runtime,
            noise_model,
            power=power,
            shots=shots,
            seed=seed,
            scale_factors=scale_factors,
            extrapolation_order=extrapolation_order,
        )

    def to_openqasm(
        self,
        *,
        power: int = 0,
        target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
        backend_mode: BackendMode | None = None,
    ) -> QasmExport:
        """Export one decomposed and routed objective circuit as OpenQASM 2."""

        return export_openqasm(
            self._portable_runtime(mode=backend_mode),
            power=power,
            target=target,
        )

    def to_qiskit(
        self,
        *,
        power: int = 0,
        target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
        backend_mode: BackendMode | None = None,
    ) -> Any:
        """Export one objective circuit to an optional Qiskit ``QuantumCircuit``."""

        return export_qiskit(
            self._portable_runtime(mode=backend_mode),
            power=power,
            target=target,
        )

    def explain(self) -> str:
        """Return a compact, human-readable compiler decision report."""
        status = "met" if self.representation_converged else "not met at max_qubits"
        successive_grid_change = (
            "not estimated"
            if self.representation.discretization_error is None
            else f"{self.representation.discretization_error:.6g}"
        )
        representation_allocation = (
            self.error_budget.domain_truncation + self.error_budget.discretization
        )
        if self.payoff_approximation is None:
            payoff_report = "Payoff: exact grid-point multiplexer"
            circuit_report = "probability-tree RY loading, multiplexed payoff rotations"
        else:
            approximation_status = "met" if self.payoff_approximation.met_tolerance else "not met"
            payoff_report = (
                f"Payoff: {self.payoff_approximation.parameter_count}/"
                f"{self.payoff_approximation.full_term_count} Walsh/Pauli terms "
                f"({100 * self.payoff_approximation.compression_ratio:.1f}% retained); "
                f"price error={self.payoff_approximation_error:.6g} "
                f"({approximation_status}; allocation={self.error_budget.algorithmic:.6g})"
            )
            circuit_report = "Hadamard quantile loading, sparse commuting Pauli payoff rotations"
        return (
            f"QFin compiled a European {self.instrument.kind} under Black-Scholes.\n"
            f"Terminal model: lognormal(mu={self.distribution.mu:.6f}, "
            f"sigma={self.distribution.sigma:.6f})\n"
            f"Representation: {self.representation.qubits} data qubits, "
            f"{self.representation.grid_points} grid points, "
            f"domain=[{self.representation.lower_bound:.6f}, "
            f"{self.representation.upper_bound:.6f}]\n"
            f"Representation validation error: {self.representation_error:.6g} "
            f"({status}; domain + discretization allocation="
            f"{representation_allocation:.6g})\n"
            f"Target error: {self.target_error:.6g} {self.target_error_unit}\n"
            f"Successive-grid price change: {successive_grid_change}\n"
            f"Encoding: {self.representation.encoding_method}; "
            f"state preparation={self.representation.state_preparation_method}\n"
            f"Selection: {self.state_preparation_strategy.selection_reason}\n"
            f"{payoff_report}\n"
            f"Circuit: {circuit_report}, gate-level reflections\n"
            f"Backend: {self.backend_name}; algorithm: MLAE; "
            f"payoff scale={self.payoff_scale:.6f}\n"
            f"Compilation converged: {self.compilation_converged}; "
            "sampling error: not estimated before execution.\n"
            "Resources: logical circuit counts, not a hardware-runtime estimate; "
            "experimental simulator workflow, no quantum-advantage claim.\n"
            f"Discrete price={self.discrete_value:.6f}; "
            f"compiled-circuit price={self.circuit_value:.6f}; "
            f"Black-Scholes={self.classical_value:.6f}"
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
        device_name: str = "auto",
    ) -> PricingResult:
        """Execute MLAE and return a validated financial result."""
        powers, shot_count = _validated_schedule(schedule, shots)
        resolved_mode = self._resolve_backend_mode(backend_mode)
        resolved_device = resolve_quantum_device(device_name)
        backend = self.to_pennylane(
            mode=resolved_mode,
            max_dense_dimension=max_dense_dimension,
            max_structured_rotations=max_structured_rotations,
            max_compressed_terms=max_compressed_terms,
            device_name=resolved_device,
        )
        observations = backend.run_schedule(powers, shots=shot_count, seed=seed)
        amplitude = maximum_likelihood_amplitude_estimate(
            observations,
            grid_size=likelihood_grid_size,
        )
        multiplier = self.discount_factor * self.payoff_scale
        value = multiplier * amplitude.amplitude
        lower = multiplier * amplitude.lower_95
        upper = multiplier * amplitude.upper_95
        absolute_error = abs(value - self.classical_value)
        reference_value = (
            self.circuit_value if resolved_mode == "compressed" else self.discrete_value
        )
        payoff_approximation_error = abs(reference_value - self.discrete_value)
        estimation_error = abs(value - reference_value)
        resources = self.resources(
            schedule=powers,
            shots=shot_count,
            backend_mode=resolved_mode,
            device_name=resolved_device,
        )
        return PricingResult(
            value=value,
            confidence_interval_95=(min(lower, upper), max(lower, upper)),
            classical_value=self.classical_value,
            discrete_value=self.discrete_value,
            circuit_value=reference_value,
            absolute_error=absolute_error,
            representation_error=self.representation_error,
            payoff_approximation_error=payoff_approximation_error,
            estimation_error=estimation_error,
            target_error=self.target_error,
            target_error_unit=self.target_error_unit,
            meets_target_error=absolute_error <= self.target_error,
            amplitude=amplitude,
            resources=resources,
            payoff_approximation=(
                self.payoff_approximation if resolved_mode == "compressed" else None
            ),
            backend=f"pennylane.{resolved_device}:{resolved_mode}",
            algorithm=self.algorithm_name,
        )
