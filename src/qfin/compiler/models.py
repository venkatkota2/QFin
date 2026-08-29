"""Compiled model and result objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
from numpy.typing import NDArray

from qfin.algorithms import AmplitudeEstimate, maximum_likelihood_amplitude_estimate
from qfin.backends import (
    CompressedPennyLaneBackend,
    DensePennyLaneBackend,
    StructuredPennyLaneBackend,
)
from qfin.circuits import WalshPayoffApproximation
from qfin.finance import BlackScholes, EuropeanOption, LogNormal
from qfin.finance.fixed_income import Engine
from qfin.finance.risk import CVaR, RiskSummary, aggregate_risk
from qfin.representation import DistributionEncoding
from qfin.resources import BackendMode, ResourceReport, estimate_resources

PennyLaneRuntime = (
    CompressedPennyLaneBackend | StructuredPennyLaneBackend | DensePennyLaneBackend
)


def _resolve_quantum_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    return (
        "lightning.qubit"
        if find_spec("pennylane_lightning") is not None
        else "default.qubit"
    )


@dataclass(frozen=True, slots=True)
class CompiledRiskModel:
    """Classical CVaR execution plus a quantum-ready loss representation.

    QFin does not yet claim a quantum CVaR algorithm. ``run`` is explicitly a
    classical NumPy/C++ aggregation, while ``representation`` is the bridge to
    future quantum tail-risk work.
    """

    problem: CVaR
    representation: DistributionEncoding
    target_error: float
    backend_name: str = "classical"
    algorithm_name: str = "weighted_discrete_expected_shortfall"
    quantum_algorithm_available: bool = False

    def run(self, *, engine: Engine = "auto") -> RiskSummary:
        return aggregate_risk(
            self.problem.distribution,
            confidence=self.problem.confidence,
            engine=engine,
        )

    def to_pennylane(self) -> None:
        from qfin.exceptions import CompilationError

        raise CompilationError(
            "QFin can encode this loss distribution, but a quantum VaR/CVaR "
            "oracle and estimator are not implemented yet"
        )

    def explain(self) -> str:
        return (
            "QFin compiled a finite CVaR problem for classical execution and "
            f"distribution encoding ({self.representation.qubits} qubits, "
            f"confidence={self.problem.confidence:.6f}). The classical weighted-tail "
            "algorithm is available; the quantum CVaR algorithm is intentionally "
            "reported as unavailable."
        )


@dataclass(frozen=True, slots=True)
class ErrorBudget:
    """Requested financial-unit tolerance split across compiler stages."""

    total: float
    domain_truncation: float
    discretization: float
    algorithmic: float
    sampling: float

    @classmethod
    def allocate(cls, target_error: float) -> ErrorBudget:
        return cls(
            total=target_error,
            domain_truncation=0.15 * target_error,
            discretization=0.35 * target_error,
            algorithmic=0.25 * target_error,
            sampling=0.25 * target_error,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "domain_truncation": self.domain_truncation,
            "discretization": self.discretization,
            "algorithmic": self.algorithmic,
            "sampling": self.sampling,
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
    meets_target_error: bool
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
            "discrete_value": self.discrete_value,
            "circuit_value": self.circuit_value,
            "absolute_error": self.absolute_error,
            "representation_error": self.representation_error,
            "payoff_approximation_error": self.payoff_approximation_error,
            "estimation_error": self.estimation_error,
            "target_error": self.target_error,
            "meets_target_error": self.meets_target_error,
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
    backend_name: str = "pennylane"
    algorithm_name: str = "maximum_likelihood_amplitude_estimation"

    @property
    def compilation_converged(self) -> bool:
        payoff_converged = (
            self.payoff_approximation is None
            or self.payoff_approximation.met_tolerance
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
        resolved_device = _resolve_quantum_device(device_name)
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
        resolved_device = _resolve_quantum_device(device_name)
        if resolved_mode == "compressed":
            if self.payoff_approximation is None:
                raise ValueError(
                    "compressed backend requires representation_method='quantile'"
                )
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

    def to_qiskit(self) -> None:
        """Make the planned but unavailable backend boundary explicit."""
        raise NotImplementedError("Qiskit export is planned after the PennyLane MVP")

    def explain(self) -> str:
        """Return a compact, human-readable compiler decision report."""
        status = "met" if self.representation_converged else "not met at max_qubits"
        representation_allocation = (
            self.error_budget.domain_truncation + self.error_budget.discretization
        )
        if self.payoff_approximation is None:
            payoff_report = "Payoff: exact grid-point multiplexer"
            circuit_report = "probability-tree RY loading, multiplexed payoff rotations"
        else:
            approximation_status = (
                "met" if self.payoff_approximation.met_tolerance else "not met"
            )
            payoff_report = (
                f"Payoff: {self.payoff_approximation.parameter_count}/"
                f"{self.payoff_approximation.full_term_count} Walsh/Pauli terms "
                f"({100 * self.payoff_approximation.compression_ratio:.1f}% retained); "
                f"price error={self.payoff_approximation_error:.6g} "
                f"({approximation_status}; allocation={self.error_budget.algorithmic:.6g})"
            )
            circuit_report = (
                "Hadamard quantile loading, sparse commuting Pauli payoff rotations"
            )
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
            f"Successive-grid price change: "
            f"{self.representation.discretization_error:.6g}\n"
            f"Encoding: {self.representation.encoding_method}; "
            f"state preparation={self.representation.state_preparation_method}\n"
            f"{payoff_report}\n"
            f"Circuit: {circuit_report}, gate-level reflections\n"
            f"Algorithm: MLAE on PennyLane; payoff scale={self.payoff_scale:.6f}\n"
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
        powers = tuple(int(power) for power in schedule)
        resolved_mode = self._resolve_backend_mode(backend_mode)
        resolved_device = _resolve_quantum_device(device_name)
        backend = self.to_pennylane(
            mode=resolved_mode,
            max_dense_dimension=max_dense_dimension,
            max_structured_rotations=max_structured_rotations,
            max_compressed_terms=max_compressed_terms,
            device_name=resolved_device,
        )
        observations = backend.run_schedule(powers, shots=shots, seed=seed)
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
            shots=shots,
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
            meets_target_error=absolute_error <= self.target_error,
            amplitude=amplitude,
            resources=resources,
            payoff_approximation=(
                self.payoff_approximation if resolved_mode == "compressed" else None
            ),
            backend=f"pennylane.{resolved_device}:{resolved_mode}",
            algorithm=self.algorithm_name,
        )
