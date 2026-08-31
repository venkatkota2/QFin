"""Compiled classical model for portfolio-optimization problems."""

from __future__ import annotations

from dataclasses import dataclass

from qfin.exceptions import CompilationError
from qfin.finance.optimization import (
    MeanVarianceProblem,
    OptimizationMethod,
    PortfolioOptimizationResult,
)
from qfin.representation.feasibility import (
    BlockEncodingFeasibility,
    analyze_block_encoding,
)
from qfin.resources.optimization import (
    OptimizationResourceReport,
    estimate_optimization_resources,
)


@dataclass(frozen=True, slots=True)
class CompiledOptimizationModel:
    """Compiler-selected continuous classical optimization workflow."""

    problem: MeanVarianceProblem
    backend_name: str = "classical"
    algorithm_name: str = "classical_continuous_mean_variance"
    quantum_representation_available: bool = False
    quantum_algorithm_available: bool = False

    def run(self, *, method: OptimizationMethod = "auto") -> PortfolioOptimizationResult:
        return self.problem.solve(method=method)

    def resources(self) -> OptimizationResourceReport:
        return estimate_optimization_resources(self.problem)

    def block_encoding_feasibility(self) -> BlockEncodingFeasibility:
        """Analyze covariance preconditions without constructing an oracle."""

        return analyze_block_encoding(self.problem.covariance)

    def to_pennylane(self) -> None:
        raise CompilationError(
            "portfolio optimization has no implemented quantum algorithm in QFin 0.9"
        )

    def explain(self) -> str:
        feasibility = self.block_encoding_feasibility()
        return (
            f"QFin classified a {self.problem.asset_count}-asset continuous mean-variance "
            "problem.\n"
            "Backend: classical SciPy SLSQP with analytical gradient for constrained "
            "problems, or the equality-constrained closed form when applicable.\n"
            "Quantum representation: unavailable; no QUBO or variational circuit is "
            "implemented.\n"
            f"Covariance is Hermitian={feasibility.hermitian} and "
            f"PSD={feasibility.positive_semidefinite}, but block encoding and QSVT "
            "remain feasibility metadata only."
        )


__all__ = ["CompiledOptimizationModel"]
