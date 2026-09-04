from datetime import date

import numpy as np
import pytest

import qfin


def test_dated_bond_schedule_accrual_and_clean_dirty_prices() -> None:
    bond = qfin.FixedRateBond(
        coupon_rate=0.05,
        issue_date="2024-01-31",
        maturity_date="2025-01-31",
        frequency=2,
        day_count="30/360",
        end_of_month=True,
    )
    assert bond.is_dated
    assert bond.payment_dates == (date(2024, 7, 31), date(2025, 1, 31))
    times, amounts = bond.cashflows(settlement="2024-04-30")
    np.testing.assert_allclose(amounts, [2.5, 102.5])
    np.testing.assert_allclose(times, [0.25, 0.75])
    assert bond.accrued_interest("2024-04-30") == pytest.approx(1.25)

    curve = qfin.YieldCurve(
        [0.0, 2.0],
        [0.0, 0.0],
        valuation_date="2024-04-30",
        day_count="ACT/365 Fixed",
    )
    priced = qfin.price_bonds(bond, curve, engine="numpy")
    assert priced.dirty_prices[0] == pytest.approx(105.0)
    assert priced.clean_prices[0] == pytest.approx(103.75)
    assert priced.accrued_interest[0] == pytest.approx(1.25)


def test_dated_stub_coupon_amounts_follow_day_count_boundaries() -> None:
    bond = qfin.FixedRateBond.from_dates(
        "2024-01-15",
        "2025-01-15",
        0.05,
        frequency=2,
        first_coupon_date="2024-04-15",
        next_to_last_coupon_date="2024-10-15",
    )
    assert bond.schedule.unadjusted_dates == (
        date(2024, 1, 15),
        date(2024, 4, 15),
        date(2024, 10, 15),
        date(2025, 1, 15),
    )
    _, amounts = bond.cashflows()
    np.testing.assert_allclose(amounts, [1.25, 2.5, 101.25])


def test_dated_yield_round_trip_and_settlement_contract() -> None:
    bond = qfin.FixedRateBond.from_dates(
        "2024-01-31",
        "2029-01-31",
        0.04,
        frequency=2,
        end_of_month=True,
    )
    priced = qfin.price_bonds_from_yield(
        bond, 0.0475, settlement="2024-04-30", engine="numpy"
    )
    solved = qfin.yield_from_prices(
        bond,
        priced.dirty_prices,
        settlement="2024-04-30",
        engine="numpy",
    )
    assert solved.converged[0]
    assert solved.yields[0] == pytest.approx(0.0475, abs=1.0e-11)
    assert priced.methodology == "yield_to_maturity"
    assert priced.ytm_modified_duration[0] == priced.modified_duration[0]
    assert np.isnan(priced.parallel_zero_duration[0])

    mismatched_curve = qfin.YieldCurve(
        [0.0, 10.0], [0.03, 0.03], valuation_date="2024-05-01"
    )
    with pytest.raises(ValueError, match="must equal"):
        qfin.price_bonds(bond, mismatched_curve, settlement="2024-04-30")


def test_curve_and_spread_risk_names_are_explicit() -> None:
    curve = qfin.YieldCurve([0.0, 2.0, 5.0, 10.0], [0.02, 0.025, 0.03, 0.035])
    bonds = [qfin.FixedRateBond(5.0, 0.0), qfin.FixedRateBond(8.0, 0.04)]
    analytics = qfin.price_bonds(bonds, curve, z_spread=0.01, engine="numpy")
    assert analytics.methodology == "curve_parallel_zero"
    np.testing.assert_allclose(analytics.parallel_zero_duration, analytics.macaulay_duration)
    np.testing.assert_allclose(analytics.spread_duration, analytics.parallel_zero_duration)
    np.testing.assert_allclose(analytics.cs01, analytics.dv01)
    assert np.all(analytics.effective_duration > 0)
    assert np.all(analytics.effective_convexity > 0)
    assert np.all(np.isnan(analytics.ytm_macaulay_duration))


def test_key_rate_risk_reconciles_to_parallel_risk() -> None:
    curve = qfin.YieldCurve([0.0, 2.0, 5.0, 10.0], [0.02, 0.025, 0.03, 0.035])
    bonds = [qfin.FixedRateBond(5.0, 0.0), qfin.FixedRateBond(8.0, 0.04)]
    report = qfin.key_rate_risk(bonds, curve, engine="numpy")
    assert report.key_rate_dv01.shape == (2, 4)
    assert report.key_rate_duration.shape == (2, 4)
    np.testing.assert_allclose(
        np.sum(report.key_rate_dv01, axis=1), report.parallel_dv01, rtol=2.0e-6
    )
    assert report.interpolation == "linear_zero"


def test_par_yield_reprices_dated_bond_to_face_value() -> None:
    curve = qfin.YieldCurve(
        [0.0, 2.0, 10.0],
        [0.03, 0.035, 0.04],
        valuation_date="2024-04-30",
    )
    template = qfin.FixedRateBond.from_dates(
        "2024-01-31",
        "2029-01-31",
        0.0,
        frequency=2,
        end_of_month=True,
    )
    coupon = qfin.par_yield(template, curve)
    par_bond = qfin.FixedRateBond.from_dates(
        "2024-01-31",
        "2029-01-31",
        coupon,
        frequency=2,
        end_of_month=True,
    )
    result = qfin.price_bonds(par_bond, curve, engine="numpy")
    assert result.clean_prices[0] == pytest.approx(100.0, abs=1.0e-11)


def test_dated_bond_input_validation() -> None:
    with pytest.raises(ValueError, match="together"):
        qfin.FixedRateBond(coupon_rate=0.03, issue_date="2024-01-01")
    with pytest.raises(ValueError, match="instead"):
        qfin.FixedRateBond(
            2.0,
            0.03,
            issue_date="2024-01-01",
            maturity_date="2026-01-01",
        )
    with pytest.raises(TypeError, match="must be a date"):
        qfin.FixedRateBond.from_dates("2024-01-01", "2026-01-01", 0.03).cashflows(
            settlement=0.5
        )
