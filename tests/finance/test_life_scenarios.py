import numpy as np
import pytest

import qfin


def _book() -> tuple[qfin.PolicyModelPointSet, qfin.ProjectionAssumptions]:
    mortality = qfin.MortalityTable([0, 120], [0.01, 0.01])
    curve = qfin.YieldCurve([0, 1, 5, 10], [0.02, 0.025, 0.03, 0.035])
    assumptions = qfin.ProjectionAssumptions(
        mortality,
        curve,
        lapse_rate=0.03,
        expense_per_policy=10,
        disability_rate=0.02,
        recovery_rate=0.10,
        disabled_mortality_multiplier=2.0,
        crediting_rate=0.03,
        expense_inflation_rate=0.02,
    )
    points = qfin.PolicyModelPointSet(
        [
            qfin.LifePolicy(40, 100_000, 500, 5),
            qfin.LifePolicy(
                65,
                0,
                0,
                5,
                product_type="annuity",
                annual_benefit=10_000,
                benefit_inflation_linkage=1,
            ),
        ],
        [100, 50],
    )
    return points, assumptions


def test_zero_factor_path_matches_base_projection() -> None:
    points, assumptions = _book()
    scenarios = qfin.EconomicScenarioSet(np.zeros((1, 5, assumptions.curve.times.size)))
    base = qfin.project_liabilities(points, assumptions, engine="numpy")
    scenario = qfin.project_liability_scenarios(points, assumptions, scenarios, engine="numpy")
    assert scenario.present_values[0] == pytest.approx(base.present_value, abs=1e-9)
    assert scenario.expected_premiums[0] == pytest.approx(np.sum(base.expected_premiums))


def test_chunked_life_scenarios_match_native_and_feed_risk() -> None:
    points, assumptions = _book()
    rates = np.zeros((6, 5, assumptions.curve.times.size))
    rates[:] = np.linspace(-0.01, 0.01, 6)[:, None, None]
    scenarios = qfin.EconomicScenarioSet(
        rates,
        mortality_multipliers=np.linspace(0.8, 1.2, 6)[:, None] * np.ones((6, 5)),
        lapse_multipliers=np.linspace(0.5, 1.5, 6)[:, None] * np.ones((6, 5)),
        inflation_rates=np.linspace(0.0, 0.03, 6)[:, None] * np.ones((6, 5)),
        probabilities=np.arange(1, 7),
    )
    reference = qfin.project_liability_scenarios(
        points,
        assumptions,
        scenarios,
        engine="numpy",
        scenario_chunk_size=2,
        policy_chunk_size=1,
    )
    native = qfin.project_liability_scenarios(
        points,
        assumptions,
        scenarios,
        engine="native",
        scenario_chunk_size=4,
        policy_chunk_size=2,
    )
    for name in (
        "present_values",
        "expected_premiums",
        "expected_benefits",
        "expected_expenses",
        "expected_surrenders",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=1e-13, atol=2e-9
        )
    losses = native.loss_distribution()
    np.testing.assert_allclose(losses.probabilities, scenarios.probabilities)
    assert native.working_set_estimate_bytes > 0


def test_life_sensitivity_report_has_financially_consistent_signs() -> None:
    points, assumptions = _book()
    report = qfin.life_sensitivities(points, assumptions, engine="native")
    assert report.mortality_impact > 0
    assert report.rate_impact < 0
    assert report.expense_impact > 0
    assert report.to_dict()["engine"] == "native"


def test_life_scenario_horizon_and_period_validation() -> None:
    points, assumptions = _book()
    short = qfin.EconomicScenarioSet(np.zeros((1, 4, assumptions.curve.times.size)))
    with pytest.raises(ValueError, match="full policy horizon"):
        qfin.project_liability_scenarios(points, assumptions, short)
    quarterly = qfin.EconomicScenarioSet(
        np.zeros((1, 5, assumptions.curve.times.size)), period_length=0.25
    )
    with pytest.raises(ValueError, match="period_length=1"):
        qfin.project_liability_scenarios(points, assumptions, quarterly)


def test_scenario_life_auto_dispatch_and_empty_book() -> None:
    points, assumptions = _book()
    scenarios = qfin.EconomicScenarioSet(np.zeros((1, 5, assumptions.curve.times.size)))
    automatic = qfin.project_liability_scenarios(points, assumptions, scenarios)
    assert automatic.engine == "native"

    empty = qfin.project_liability_scenarios([], assumptions, scenarios, engine="native")
    np.testing.assert_array_equal(empty.present_values, np.zeros(1))
    assert empty.model_point_count == 0
    assert empty.policy_count == 0
