"""Honest classical and future-quantum optimization resource metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qfin.finance.optimization import MeanVarianceProblem


@dataclass(frozen=True, slots=True)
class OptimizationResourceReport:
    """Resource boundary for the implemented continuous classical solver."""

    assets: int
    continuous_variables: int
    equality_constraints: int
    inequality_constraints: int
    covariance_entries: int
    classical_input_memory_bytes: int
    classical_solver: str
    classical_complexity: str
    quantum_representation_available: bool = False
    quantum_algorithm_available: bool = False
    binary_encoding_qubits: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "assets": self.assets,
            "continuous_variables": self.continuous_variables,
            "equality_constraints": self.equality_constraints,
            "inequality_constraints": self.inequality_constraints,
            "covariance_entries": self.covariance_entries,
            "classical_input_memory_bytes": self.classical_input_memory_bytes,
            "classical_solver": self.classical_solver,
            "classical_complexity": self.classical_complexity,
            "quantum_representation_available": self.quantum_representation_available,
            "quantum_algorithm_available": self.quantum_algorithm_available,
            "binary_encoding_qubits": self.binary_encoding_qubits,
            "caveat": (
                "QFin 0.9 implements a continuous classical baseline only. No QUBO, "
                "variational optimizer, annealer, or quantum advantage is claimed."
            ),
        }


def estimate_optimization_resources(
    problem: MeanVarianceProblem,
) -> OptimizationResourceReport:
    lower = np.asarray(problem.lower_bounds, dtype=np.float64)
    upper = np.asarray(problem.upper_bounds, dtype=np.float64)
    finite_bounds = int(np.count_nonzero(np.isfinite(lower))) + int(
        np.count_nonzero(np.isfinite(upper))
    )
    return OptimizationResourceReport(
        assets=problem.asset_count,
        continuous_variables=problem.asset_count,
        equality_constraints=1,
        inequality_constraints=finite_bounds + int(problem.target_return is not None),
        covariance_entries=problem.asset_count**2,
        classical_input_memory_bytes=int(
            problem.expected_returns.nbytes
            + problem.covariance.nbytes
            + lower.nbytes
            + upper.nbytes
        ),
        classical_solver=(
            "SciPy SLSQP with analytical gradient; equality-constrained closed form when unbounded"
        ),
        classical_complexity="dense constrained optimization; approximately O(assets**3)",
    )


__all__ = ["OptimizationResourceReport", "estimate_optimization_resources"]
