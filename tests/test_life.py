import numpy as np
import pytest

import qfin


@pytest.fixture
def mortality() -> qfin.MortalityTable:
    return qfin.MortalityTable(
        ages=np.arange(30, 41),
        qx=np.array([0.10] * 10 + [1.0]),
    )


def test_mortality_survival_and_death_probabilities(
    mortality: qfin.MortalityTable,
) -> None:
    assert mortality.survival_start_probabilities(30, 3) == pytest.approx(
        [1.0, 0.9, 0.81]
    )
    assert mortality.death_probabilities(30, 3) == pytest.approx(
        [0.1, 0.09, 0.081]
    )


def test_term_life_projects_benefits_premiums_and_expenses(
    mortality: qfin.MortalityTable,
) -> None:
    policy = qfin.TermLifePolicy(
        issue_age=30,
        term=3,
        face_amount=100.0,
        annual_premium=2.0,
        premium_term=2,
        annual_expense=1.0,
    )
    projection = policy.expected_cashflows(mortality)
    assert projection.times.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert projection.benefits == pytest.approx([0.0, 10.0, 9.0, 8.1])
    assert projection.premiums == pytest.approx([2.0, 1.8, 0.0, 0.0])
    assert projection.expenses == pytest.approx([1.0, 0.9, 0.81, 0.0])
    assert projection.net_liabilities == pytest.approx([-1.0, 9.1, 9.81, 8.1])


def test_whole_life_and_portfolio_aggregation(
    mortality: qfin.MortalityTable,
) -> None:
    whole = qfin.WholeLifePolicy(
        issue_age=38,
        face_amount=200.0,
        annual_premium=3.0,
        premium_term=2,
    )
    term = qfin.TermLifePolicy(issue_age=38, term=2, face_amount=100.0)
    portfolio = qfin.LifePolicyPortfolio(
        (qfin.PolicyPosition(whole, 2.0), qfin.PolicyPosition(term, 1.0))
    )
    projection = portfolio.expected_cashflows(mortality)
    assert projection.times.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert np.all(projection.benefits >= 0)
    assert projection.benefits[-1] > 0
    assert projection.net_schedule.times.shape == projection.times.shape


def test_illustrative_table_is_explicit_and_closed() -> None:
    table = qfin.MortalityTable.illustrative_gompertz_makeham(
        min_age=20, max_age=100
    )
    assert table.qx[-1] == 1.0
    assert np.all(np.diff(table.qx) >= 0)
