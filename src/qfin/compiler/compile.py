"""Financial-to-quantum compiler entry point."""

from dataclasses import replace
from math import exp, isfinite
from typing import Literal

import numpy as np

from qfin.circuits import WalshPayoffApproximation
from qfin.compiler.models import CompiledPricingModel, CompiledRiskModel, ErrorBudget
from qfin.exceptions import CompilationError
from qfin.finance import (
    BlackScholes,
    EuropeanCall,
    EuropeanOption,
    EuropeanPut,
    GeometricBrownianMotion,
)
from qfin.finance.risk import CVaR
from qfin.representation import encode, encode_quantiles
from qfin.validation import black_scholes_price


def compile(
    problem: EuropeanOption | CVaR,
    market: BlackScholes | None = None,
    *,
    target_error: float = 0.01,
    backend: str = "auto",
    min_qubits: int = 3,
    max_qubits: int = 12,
    tail_probability: float | None = None,
    representation_method: Literal["quantile", "probability"] = "quantile",
    payoff_angle_tolerance: float = 0.1,
    payoff_max_terms: int | None = None,
) -> CompiledPricingModel | CompiledRiskModel:
    """Compile a supported financial problem into the QFin pipeline.

    ``target_error`` is stated in price units. The returned object can be
    inspected without PennyLane; PennyLane is imported only when ``run`` or
    ``to_pennylane`` executes a circuit.
    """

    if isinstance(problem, CVaR):
        if market is not None:
            raise CompilationError("CVaR compilation does not accept a BlackScholes market")
        if backend not in ("auto", "classical"):
            raise CompilationError(
                "quantum CVaR execution is not implemented; use backend='auto' or 'classical'"
            )
        if not isfinite(target_error) or target_error <= 0:
            raise ValueError("target_error must be finite and greater than zero")
        risk_representation = encode(
            problem.distribution.as_empirical(),
            target_error=target_error,
            objective="expectation",
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            tail_probability=0.0,
        )
        return CompiledRiskModel(
            problem=problem,
            representation=risk_representation,
            target_error=target_error,
        )

    if not isinstance(problem, (EuropeanCall, EuropeanPut)):
        raise CompilationError(
            "option compilation supports EuropeanCall and EuropeanPut only"
        )
    if not isinstance(market, BlackScholes):
        raise CompilationError("European options require a BlackScholes market model")
    resolved_backend = "pennylane" if backend == "auto" else backend
    if resolved_backend != "pennylane":
        raise CompilationError("option compilation supports backend='pennylane' only")
    if not isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be finite and greater than zero")
    if min_qubits < 1 or max_qubits < min_qubits:
        raise ValueError("require 1 <= min_qubits <= max_qubits")
    if representation_method not in ("quantile", "probability"):
        raise ValueError("representation_method must be 'quantile' or 'probability'")
    if not isfinite(payoff_angle_tolerance) or payoff_angle_tolerance <= 0:
        raise ValueError("payoff_angle_tolerance must be finite and positive")
    if payoff_max_terms is not None and payoff_max_terms < 1:
        raise ValueError("payoff_max_terms must be positive")

    budget = ErrorBudget.allocate(target_error)
    if tail_probability is None:
        financial_scale = max(market.spot, problem.strike, 1.0)
        tail_probability = float(
            np.clip(budget.domain_truncation / (100.0 * financial_scale), 1e-10, 1e-4)
        )
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must lie strictly between zero and one")

    discount_factor = exp(-market.rate * problem.maturity)
    distribution = GeometricBrownianMotion(market).terminal_distribution(problem.maturity)
    classical_value = black_scholes_price(problem, market)

    def discounted_payoff(grid: np.ndarray) -> np.ndarray:
        return discount_factor * problem.payoff(grid)

    encoder = encode_quantiles if representation_method == "quantile" else encode
    representation = None
    raw_payoff = None
    discrete_value = 0.0
    previous_value: float | None = None
    representation_tolerance = budget.domain_truncation + budget.discretization

    # The generic encoders use successive-grid stabilization because they do
    # not know an exact expectation. For the Black-Scholes MVP an analytical
    # benchmark is available, so use it to prevent false convergence when two
    # coarse grids both miss a low-probability, non-zero payoff region.
    for candidate in range(min_qubits, max_qubits + 1):
        candidate_representation = encoder(
            distribution,
            target_error=budget.discretization,
            objective=discounted_payoff,
            qubits=candidate,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            tail_probability=tail_probability,
        )
        candidate_payoff = problem.payoff(candidate_representation.grid)
        candidate_value = discount_factor * float(
            np.dot(candidate_representation.probabilities, candidate_payoff)
        )
        refinement_change = (
            float("inf")
            if previous_value is None
            else abs(candidate_value - previous_value)
        )
        representation = replace(
            candidate_representation,
            discretization_error=refinement_change,
        )
        raw_payoff = candidate_payoff
        discrete_value = candidate_value
        if abs(discrete_value - classical_value) <= representation_tolerance:
            break
        previous_value = candidate_value

    assert representation is not None
    assert raw_payoff is not None
    payoff_scale = float(np.max(raw_payoff))
    if payoff_scale == 0.0:
        normalized_payoff = np.zeros_like(raw_payoff)
    else:
        normalized_payoff = raw_payoff / payoff_scale
    representation_error = abs(discrete_value - classical_value)
    representation_converged = representation_error <= representation_tolerance
    payoff_approximation: WalshPayoffApproximation | None = None
    if representation_method == "quantile":
        payoff_approximation = WalshPayoffApproximation.fit(
            normalized_payoff,
            financial_multiplier=discount_factor * payoff_scale,
            target_price_error=budget.algorithmic,
            max_angle_rmse=payoff_angle_tolerance,
            max_terms=(
                None
                if payoff_max_terms is None
                else min(payoff_max_terms, normalized_payoff.size)
            ),
        )
    circuit_value = (
        discrete_value
        if payoff_approximation is None
        else discount_factor
        * payoff_scale
        * payoff_approximation.approximate_amplitude
    )
    payoff_approximation_error = abs(circuit_value - discrete_value)
    raw_payoff.setflags(write=False)
    normalized_payoff.setflags(write=False)

    return CompiledPricingModel(
        instrument=problem,
        market=market,
        distribution=distribution,
        representation=representation,
        raw_payoff=raw_payoff,
        normalized_payoff=normalized_payoff,
        payoff_scale=payoff_scale,
        discount_factor=discount_factor,
        discrete_value=discrete_value,
        classical_value=classical_value,
        target_error=target_error,
        error_budget=budget,
        representation_error=representation_error,
        representation_converged=representation_converged,
        payoff_approximation=payoff_approximation,
        circuit_value=circuit_value,
        payoff_approximation_error=payoff_approximation_error,
        representation_method=representation_method,
        backend_name=resolved_backend,
    )
