from datetime import date

import numpy as np
import pytest

import qfin


def _alm_model() -> qfin.ALMModel:
    curve = qfin.YieldCurve([0.0, 1.0, 5.0, 10.0], [0.02, 0.025, 0.03, 0.035])
    assets = qfin.AssetPortfolio(
        [
            qfin.FixedRateBond(3.0, 0.03),
            qfin.FixedRateBond(8.0, 0.04),
        ],
        [8.0, 12.0],
    )
    liabilities = qfin.LiabilityPortfolio.from_arrays(
        [2.0, 5.0, 9.0], [500.0, 700.0, 900.0]
    )
    return qfin.ALMModel(assets, liabilities, curve)


def test_alm_evaluation_fields_and_gap_definition() -> None:
    result = _alm_model().evaluate(engine="numpy")
    assert result.surplus == pytest.approx(result.asset_pv - result.liability_pv)
    assert result.funding_ratio == pytest.approx(result.asset_pv / result.liability_pv)
    assert result.duration_gap == pytest.approx(
        result.asset_duration
        - (result.liability_pv / result.asset_pv) * result.liability_duration
    )
    assert result.asset_convexity > 0
    assert result.liability_convexity > 0


def test_parallel_rate_scenarios_reprice_both_sides() -> None:
    model = _alm_model()
    scenarios = qfin.RateScenarioSet.parallel(model.curve, [-0.01, 0.0, 0.01])
    result = model.run_scenarios(scenarios, engine="numpy", chunk_size=2)
    assert result.labels[1].startswith("parallel_1_")
    assert result.asset_pv[0] > result.asset_pv[1] > result.asset_pv[2]
    assert result.liability_pv[0] > result.liability_pv[1] > result.liability_pv[2]
    assert result.asset_pv[1] == pytest.approx(model.evaluate(engine="numpy").asset_pv)
    assert result.loss_distribution().losses.shape == (3,)


def test_steepener_and_key_rate_scenario_shapes() -> None:
    curve = _alm_model().curve
    steepener = qfin.RateScenarioSet.steepener(
        curve, short_shift=-0.005, long_shift=0.01
    )
    key_rate = qfin.RateScenarioSet.key_rate(
        curve, key_time=5.0, shift=0.01, width=4.0
    )
    assert steepener.shocks.shape == key_rate.shocks.shape == (1, curve.times.size)
    assert key_rate.shocks[0, 2] == pytest.approx(0.01)


def test_single_node_curve_scenarios_are_supported() -> None:
    curve = qfin.YieldCurve([0.0], [0.03])
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(1.0, 0.0)], [1.0]),
        qfin.LiabilityPortfolio.from_arrays([1.0], [95.0]),
        curve,
    )
    scenarios = qfin.RateScenarioSet.parallel(curve, [0.0, 0.01])
    result = model.run_scenarios(scenarios, engine="numpy")
    assert np.all(np.isfinite(result.surplus))


def test_dated_bonds_flow_consistently_through_alm_and_scenario_loss() -> None:
    curve = qfin.YieldCurve(
        [0.0, 1.0, 4.0, 12.0],
        [-0.005, 0.015, 0.035, 0.025],
        interpolation="log_linear_discount",
        extrapolation="flat_forward",
        valuation_date="2024-04-30",
        day_count="ACT/365 Fixed",
    )
    bonds = [
        qfin.FixedRateBond.from_dates(
            "2024-01-31",
            "2029-01-31",
            0.04,
            frequency=2,
            end_of_month=True,
        ),
        qfin.FixedRateBond.from_dates(
            "2023-10-15",
            "2034-10-15",
            0.055,
            frequency=1,
        ),
    ]
    quantities = np.array([7.5, 2.25])
    assets = qfin.AssetPortfolio(bonds, quantities)
    liabilities = qfin.LiabilityPortfolio.from_arrays(
        [1.0, 3.0, 8.0],
        [250.0, 500.0, 900.0],
    )
    model = qfin.ALMModel(assets, liabilities, curve)

    direct = qfin.price_bonds(bonds, curve, engine="numpy")
    base = model.evaluate(engine="numpy")
    assert assets.settlement is None
    assert base.asset_pv == pytest.approx(float(quantities @ direct.dirty_prices), abs=1.0e-11)
    assert base.asset_duration == pytest.approx(
        float(quantities @ (direct.dirty_prices * direct.macaulay_duration)) / base.asset_pv,
        abs=1.0e-12,
    )
    np.testing.assert_allclose(
        direct.clean_prices + direct.accrued_interest,
        direct.dirty_prices,
        rtol=0.0,
        atol=1.0e-13,
    )
    assert direct.accrued_interest[0] > 0.0

    shocks = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.002, -0.001, 0.004, 0.007],
            [-0.01, -0.01, -0.01, -0.01],
        ]
    )
    scenario_result = model.run_scenarios(qfin.RateScenarioSet(shocks), engine="numpy")
    expected_assets = np.array(
        [
            quantities
            @ qfin.price_bonds(
                bonds,
                curve.shifted(shock),
                engine="numpy",
            ).dirty_prices
            for shock in shocks
        ]
    )
    np.testing.assert_allclose(scenario_result.asset_pv, expected_assets, rtol=2.0e-14)
    assert scenario_result.asset_pv[0] == pytest.approx(base.asset_pv, abs=1.0e-11)
    distribution = scenario_result.loss_distribution()
    np.testing.assert_allclose(
        distribution.losses,
        base.surplus - scenario_result.surplus,
        rtol=0.0,
        atol=1.0e-13,
    )

    key_rates = qfin.key_rate_risk(bonds, curve, engine="numpy")
    np.testing.assert_allclose(
        key_rates.base_prices,
        direct.dirty_prices,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_asset_portfolio_enforces_shared_settlement_contract() -> None:
    dated = qfin.FixedRateBond.from_dates("2024-01-01", "2027-01-01", 0.03)
    floating = qfin.FixedRateBond(3.0, 0.03)

    with pytest.raises(ValueError, match="dated and floating-time"):
        qfin.AssetPortfolio([dated, floating])
    with pytest.raises(TypeError, match="must be a date"):
        qfin.AssetPortfolio([dated], settlement=0.25)
    with pytest.raises(TypeError, match="must be numeric"):
        qfin.AssetPortfolio([floating], settlement="2024-04-30")

    dated_assets = qfin.AssetPortfolio([dated], settlement="2024-04-30")
    assert dated_assets.settlement == date(2024, 4, 30)
    liabilities = qfin.LiabilityPortfolio.from_arrays([1.0], [50.0])
    mismatched_curve = qfin.YieldCurve(
        [0.0, 5.0],
        [0.02, 0.03],
        valuation_date="2024-05-01",
    )
    with pytest.raises(ValueError, match="must equal"):
        qfin.ALMModel(dated_assets, liabilities, mismatched_curve)

    undated_curve = qfin.YieldCurve([0.0, 5.0], [0.02, 0.03])
    with pytest.raises(ValueError, match="requires settlement or curve valuation_date"):
        qfin.ALMModel(qfin.AssetPortfolio([dated]), liabilities, undated_curve)
