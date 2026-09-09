"""Compiler failure, budget and metadata contracts at numerical boundaries."""

from dataclasses import replace

import pytest

import qfin

_BUDGETS = [
    qfin.ErrorBudget.allocate(1), qfin.RiskErrorBudget.allocate(1),
    qfin.StructuredOracleErrorBudget.allocate(1), qfin.StructuredRiskErrorBudget.allocate(1),
]


@pytest.mark.parametrize("budget", _BUDGETS)
@pytest.mark.parametrize("change,message", [({"total": float("nan")}, "target_error"),
                                         ({"total": 2}, "sum to total"),
                                         ({"target_error_unit": " "}, "unit")])
def test_direct_budget_construction_revalidates_invariants(
    budget: object, change: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(budget, **change)


@pytest.mark.parametrize("budget,component", list(zip(_BUDGETS,
    ("sampling", "estimation", "estimation", "estimation"), strict=True)))
def test_negative_budget_component_is_rejected(budget: object, component: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        replace(budget, **{component: -1})


@pytest.mark.parametrize("budget", _BUDGETS[1:])
@pytest.mark.parametrize("level", [0, 1, float("nan")])
def test_invalid_budget_interval_level_is_rejected(budget: object, level: float) -> None:
    with pytest.raises(ValueError, match="interval_level"):
        replace(budget, interval_level=level)


@pytest.mark.parametrize("kwargs,message", [
    ({"representation_method": "unsupported"}, "representation_method"),
    ({"payoff_angle_tolerance": 0}, "payoff_angle_tolerance"),
    ({"arithmetic_scale": -1}, "arithmetic_scale"),
    ({"tail_probability": float("nan")}, "tail_probability"),
])
def test_compiler_rejects_invalid_numerical_settings(
    kwargs: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        qfin.compile(qfin.EuropeanCall(100, 1), qfin.BlackScholes(100, .03, .2), **kwargs)


def test_problem_market_and_backend_mismatches_fail_explicitly() -> None:
    market = qfin.BlackScholes(100, .03, .2)
    risk = qfin.VaR(qfin.LossDistribution([0, 1]), .95)
    with pytest.raises(qfin.CompilationError, match="does not accept"):
        qfin.compile(risk, market)
    with pytest.raises(qfin.CompilationError, match="risk compilation supports"):
        qfin.compile(risk, backend="unsupported")
    with pytest.raises(qfin.CompilationError, match="require a BlackScholes"):
        qfin.compile(qfin.EuropeanCall(100, 1))
    with pytest.raises(qfin.CompilationError, match="EuropeanCall and EuropeanPut"):
        qfin.compile(object())
    optimization = qfin.MeanVarianceProblem([.04, .08], [[.01, 0], [0, .02]])
    with pytest.raises(qfin.CompilationError, match="does not accept"):
        qfin.compile(optimization, market)
    with pytest.raises(qfin.CompilationError, match="no implemented PennyLane"):
        qfin.compile(optimization, backend="pennylane")
