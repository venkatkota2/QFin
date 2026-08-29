import numpy as np
import pytest

import qfin


def _flat_curve() -> qfin.YieldCurve:
    return qfin.YieldCurve([0.0, 50.0], [0.0, 0.0])


def test_zero_mortality_zero_lapse_cashflows() -> None:
    mortality = qfin.MortalityTable([0, 120], [0, 0])
    policy = qfin.LifePolicy(age=40, sum_assured=100_000, annual_premium=100, term=3)
    assumptions = qfin.ProjectionAssumptions(
        mortality, _flat_curve(), lapse_rate=0.0, expense_per_policy=10
    )
    result = qfin.project_liabilities([policy], assumptions, engine="numpy")
    np.testing.assert_allclose(result.expected_premiums, [100, 100, 100, 0])
    np.testing.assert_allclose(result.expected_benefits, 0)
    np.testing.assert_allclose(result.expected_expenses, [10, 10, 10, 0])
    np.testing.assert_allclose(result.net_liability_cashflows, [-90, -90, -90, 0])
    assert result.present_value == pytest.approx(-270)


def test_certain_first_year_death_and_high_lapse() -> None:
    mortality = qfin.MortalityTable([0, 120], [1, 1])
    policy = qfin.LifePolicy(age=40, sum_assured=1_000, annual_premium=100, term=5)
    assumptions = qfin.ProjectionAssumptions(
        mortality, _flat_curve(), lapse_rate=1.0, expense_per_policy=0
    )
    result = qfin.project_liabilities([policy], assumptions, engine="numpy")
    np.testing.assert_allclose(result.expected_premiums, [100, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(result.expected_benefits, [0, 1_000, 0, 0, 0, 0])
    assert result.present_value == pytest.approx(900)


def test_policy_duration_and_mortality_category_validation() -> None:
    policy = qfin.LifePolicy(
        age=45,
        issue_age=40,
        policy_duration=5,
        term=20,
        sum_assured=10_000,
        annual_premium=50,
        mortality_category="male",
    )
    assert policy.remaining_term == 15
    assumptions = qfin.ProjectionAssumptions(
        qfin.MortalityTable([0, 120], [0.01, 0.01], category="female"),
        _flat_curve(),
    )
    with pytest.raises(ValueError, match="category"):
        qfin.project_liabilities([policy], assumptions)
    with pytest.raises(ValueError, match="issue_age"):
        qfin.LifePolicy(44, 10_000, 50, 20, issue_age=40, policy_duration=5)
    with pytest.raises(ValueError, match="integers"):
        qfin.LifePolicy(40, 10_000, 50, 20.5)  # type: ignore[arg-type]


def test_projection_converts_to_alm_liability_portfolio() -> None:
    mortality = qfin.MortalityTable([0, 120], [0.01, 0.01])
    assumptions = qfin.ProjectionAssumptions(mortality, _flat_curve(), expense_per_policy=5)
    result = qfin.project_liabilities(
        [qfin.LifePolicy(40, 10_000, 20, 2)], assumptions, engine="numpy"
    )
    portfolio = result.to_liability_portfolio()
    assert len(portfolio.cashflows) > 0
    assert sum(item.amount for item in portfolio.cashflows) == pytest.approx(
        np.sum(result.net_liability_cashflows)
    )
