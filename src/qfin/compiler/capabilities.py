"""Honest implementation metadata for financial problem categories."""

from __future__ import annotations

from dataclasses import dataclass

from qfin import _native
from qfin.finance import EuropeanCall, EuropeanPut
from qfin.finance.alm import ALMModel, ALMScenarioResult
from qfin.finance.life import LifePolicy
from qfin.finance.risk import CVaR, LossDistribution, TailProbability, VaR


@dataclass(frozen=True, slots=True)
class ProblemCapabilities:
    """Separate financial, classical, representation, and algorithm support."""

    category: str
    financial_model_available: bool
    classical_implementation: str
    native_implementation_available: bool
    quantum_representation_available: bool
    quantum_algorithm_available: bool
    note: str

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "category": self.category,
            "financial_model_available": self.financial_model_available,
            "classical_implementation": self.classical_implementation,
            "native_implementation_available": self.native_implementation_available,
            "quantum_representation_available": self.quantum_representation_available,
            "quantum_algorithm_available": self.quantum_algorithm_available,
            "note": self.note,
        }


def problem_capabilities(problem: object) -> ProblemCapabilities:
    """Report implemented boundaries without implying unsupported quantum paths."""

    if isinstance(problem, (EuropeanCall, EuropeanPut)):
        return ProblemCapabilities(
            category="derivative_pricing",
            financial_model_available=True,
            classical_implementation="Black-Scholes analytical validation",
            native_implementation_available=False,
            quantum_representation_available=True,
            quantum_algorithm_available=True,
            note="European option pricing is implemented with PennyLane MLAE.",
        )
    if isinstance(problem, (TailProbability, VaR, CVaR)):
        return ProblemCapabilities(
            category="tail_risk",
            financial_model_available=True,
            classical_implementation="weighted finite-distribution VaR/CVaR",
            native_implementation_available=_native.available(),
            quantum_representation_available=True,
            quantum_algorithm_available=True,
            note=(
                "Experimental PennyLane MLAE tail objectives are implemented. VaR "
                "uses hybrid binary search; CVaR uses the selected VaR plus a "
                "tail-excess amplitude. Generic loading is O(2**qubits)."
            ),
        )
    if isinstance(problem, (LossDistribution, ALMScenarioResult)):
        return ProblemCapabilities(
            category="tail_risk_input",
            financial_model_available=True,
            classical_implementation="weighted finite loss distribution",
            native_implementation_available=_native.available(),
            quantum_representation_available=True,
            quantum_algorithm_available=False,
            note="Wrap the loss distribution in TailProbability, VaR, or CVaR.",
        )
    if isinstance(problem, ALMModel):
        return ProblemCapabilities(
            category="asset_liability_management",
            financial_model_available=True,
            classical_implementation="curve valuation and batched rate scenarios",
            native_implementation_available=_native.available(),
            quantum_representation_available=False,
            quantum_algorithm_available=False,
            note="Run scenarios first to produce a representable loss distribution.",
        )
    if isinstance(problem, LifePolicy):
        return ProblemCapabilities(
            category="life_liability_projection",
            financial_model_available=True,
            classical_implementation="annual term-life expected cash-flow projection",
            native_implementation_available=_native.available(),
            quantum_representation_available=False,
            quantum_algorithm_available=False,
            note="Projection outputs can feed ALM; no direct quantum algorithm is claimed.",
        )
    return ProblemCapabilities(
        category="unsupported",
        financial_model_available=False,
        classical_implementation="unavailable",
        native_implementation_available=False,
        quantum_representation_available=False,
        quantum_algorithm_available=False,
        note="No compiler policy is registered for this problem type.",
    )


__all__ = ["ProblemCapabilities", "problem_capabilities"]
