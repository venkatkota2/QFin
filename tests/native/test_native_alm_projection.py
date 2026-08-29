import numpy as np
import pytest

import qfin


def test_native_alm_scenarios_match_numpy_with_chunking() -> None:
    curve = qfin.YieldCurve([0, 1, 5, 10, 30], [0.02, 0.025, 0.03, 0.035, 0.04])
    bonds = [qfin.FixedRateBond(1 + index % 20, 0.04) for index in range(100)]
    model = qfin.ALMModel(
        qfin.AssetPortfolio(bonds, np.linspace(1, 5, len(bonds))),
        qfin.LiabilityPortfolio.from_arrays(
            np.arange(1, 31, dtype=float), np.linspace(100, 500, 30)
        ),
        curve,
    )
    shifts = np.linspace(-0.02, 0.02, 257)
    scenarios = qfin.RateScenarioSet.parallel(curve, shifts)
    reference = model.run_scenarios(scenarios, engine="numpy", chunk_size=31)
    native = model.run_scenarios(scenarios, engine="native", chunk_size=37)
    np.testing.assert_allclose(native.asset_pv, reference.asset_pv, rtol=1e-13)
    np.testing.assert_allclose(native.liability_pv, reference.liability_pv, rtol=1e-13)


def test_native_policy_projection_matches_python_oracle() -> None:
    ages = np.arange(20, 121, dtype=float)
    mortality = qfin.MortalityTable(ages, np.minimum(0.0002 * np.exp(0.075 * (ages - 20)), 1))
    curve = qfin.YieldCurve([0, 1, 5, 10, 30, 50], [0.02, 0.022, 0.027, 0.03, 0.035, 0.037])
    policies = [
        qfin.LifePolicy(
            age=30 + index % 40,
            sum_assured=50_000 + 1_000 * (index % 20),
            annual_premium=200 + index % 100,
            term=10 + index % 20,
        )
        for index in range(1_000)
    ]
    assumptions = qfin.ProjectionAssumptions(
        mortality,
        curve,
        lapse_rate=np.linspace(0.08, 0.02, 30),
        expense_per_policy=25,
        mortality_multiplier=1.1,
    )
    reference = qfin.project_liabilities(policies, assumptions, engine="numpy")
    native = qfin.project_liabilities(policies, assumptions, engine="native")
    np.testing.assert_allclose(
        native.expected_premiums, reference.expected_premiums, rtol=1e-13
    )
    np.testing.assert_allclose(
        native.expected_benefits, reference.expected_benefits, rtol=1e-13
    )
    np.testing.assert_allclose(
        native.net_liability_cashflows,
        reference.net_liability_cashflows,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        native.policy_present_values,
        reference.policy_present_values,
        rtol=1e-13,
    )


def test_native_binding_rejects_malformed_buffers_without_unsafe_access() -> None:
    native = qfin._native.require()
    with pytest.raises(ValueError, match="one-dimensional"):
        native.price_cashflow_batches(
            np.array([[1.0]]),
            np.array([100.0]),
            np.array([0, 1], dtype=np.int64),
            np.array([0.0, 1.0]),
            np.array([0.02, 0.02]),
            0.0,
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        native.scenario_portfolio_present_values(
            np.array([1.0]),
            np.array([100.0]),
            np.array([0, 2, 1], dtype=np.int64),
            np.array([1.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.02, 0.02]),
            np.zeros((1, 2)),
        )
