from datetime import date
from decimal import Decimal, localcontext

import numpy as np
import pytest

import qfin
from qfin.finance.fixed_income import Settlement


def _with_coupon_rate(bond: qfin.FixedRateBond, coupon_rate: float) -> qfin.FixedRateBond:
    if not bond.is_dated:
        return qfin.FixedRateBond(
            bond.maturity,
            coupon_rate,
            face_value=bond.face_value,
            frequency=bond.frequency,
            day_count=bond.day_count,
        )
    assert bond.issue_date is not None
    assert bond.maturity_date is not None
    return qfin.FixedRateBond.from_dates(
        bond.issue_date,
        bond.maturity_date,
        coupon_rate,
        face_value=bond.face_value,
        frequency=bond.frequency,
        day_count=bond.day_count,
        calendar=bond.calendar,
        business_day_convention=bond.business_day_convention,
        termination_convention=bond.termination_convention,
        date_generation=bond.date_generation,
        end_of_month=bond.end_of_month,
        first_coupon_date=bond.first_coupon_date,
        next_to_last_coupon_date=bond.next_to_last_coupon_date,
    )


def _two_repricing_par_yield_oracle(
    bond: qfin.FixedRateBond,
    curve: qfin.YieldCurve,
    *,
    settlement: Settlement = None,
    target_clean_price: float,
) -> float:
    zero = qfin.price_bonds(
        _with_coupon_rate(bond, 0.0), curve, settlement=settlement, engine="numpy"
    ).clean_prices[0]
    unit = qfin.price_bonds(
        _with_coupon_rate(bond, 1.0), curve, settlement=settlement, engine="numpy"
    ).clean_prices[0]
    return float((target_clean_price - zero) / (unit - zero))


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


@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_tiny_key_rate_exposure_matches_high_precision_oracle(engine: str) -> None:
    if engine == "native" and not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    maturity = 1.0
    upper_node = float(np.nextafter(1.0, 2.0))
    curve = qfin.YieldCurve([0.5, upper_node, 2.0], [0.01, 0.02, 0.03])
    bond = qfin.FixedRateBond(maturity, 0.0, face_value=1.0e12)
    bump = 1.0e-4
    with localcontext() as context:
        context.prec = 70
        time = Decimal.from_float(maturity)
        upper = Decimal.from_float(upper_node)
        weight = (upper - time) / (upper - Decimal("0.5"))
        rate = weight * Decimal.from_float(0.01) + (1 - weight) * Decimal.from_float(0.02)
        shock = weight * Decimal.from_float(bump)
        down = Decimal("1e12") * (-(rate - shock) * time).exp()
        up = Decimal("1e12") * (-(rate + shock) * time).exp()
        expected = float((down - up) / 2)
    result = qfin.key_rate_risk(bond, curve, engine=engine)
    assert 0.0 < expected < 1.0e-6
    assert result.key_rate_dv01[0, 0] == pytest.approx(expected, rel=3.0e-14, abs=0.0)


@pytest.mark.parametrize("face", [1.0e-20, 1.0, 100.0, 1.0e12])
def test_par_yield_is_independent_of_notional_scale(face: float) -> None:
    curve = qfin.YieldCurve([0.0, 5.0], [0.025, 0.025])
    bond = qfin.FixedRateBond(5.0, 0.0, face_value=face, frequency=2)
    expected = 2.0 * np.expm1(0.025 / 2.0)
    assert qfin.par_yield(bond, curve) == pytest.approx(expected, rel=2.0e-14, abs=0.0)


@pytest.mark.parametrize(
    "interpolation",
    ["linear_zero", "linear_discount", "log_linear_discount", "monotone_zero"],
)
@pytest.mark.parametrize("extrapolation", ["flat_zero", "flat_forward"])
def test_prepared_key_rate_execution_matches_brute_force_repricing(
    interpolation: str,
    extrapolation: str,
) -> None:
    curve = qfin.YieldCurve(
        [0.5, 2.0, 7.0, 30.0],
        [-0.015, 0.02, 0.065, 0.03],
        interpolation=interpolation,
        extrapolation=extrapolation,
    )
    bonds = [
        qfin.FixedRateBond(0.1, 0.0),
        qfin.FixedRateBond(5.25, 0.045, frequency=4),
        qfin.FixedRateBond(55.0, 0.01, frequency=1),
    ]
    bump = 2.5e-5

    actual = qfin.key_rate_risk(bonds, curve, bump_size=bump, engine="numpy")
    base = qfin.price_bonds(bonds, curve, engine="numpy").dirty_prices
    expected = np.empty_like(actual.key_rate_dv01)
    for node in range(curve.times.size):
        shock = np.zeros(curve.times.size)
        shock[node] = bump
        up = qfin.price_bonds(bonds, curve.shifted(shock), engine="numpy").dirty_prices
        down = qfin.price_bonds(bonds, curve.shifted(-shock), engine="numpy").dirty_prices
        expected[:, node] = (down - up) / (2.0 * bump) * 1.0e-4
    parallel_up = qfin.price_bonds(
        bonds,
        curve.shifted(bump),
        engine="numpy",
    ).dirty_prices
    parallel_down = qfin.price_bonds(
        bonds,
        curve.shifted(-bump),
        engine="numpy",
    ).dirty_prices

    np.testing.assert_allclose(actual.base_prices, base, rtol=2.0e-14, atol=1.0e-12)
    np.testing.assert_allclose(actual.key_rate_dv01, expected, rtol=2.0e-10, atol=2.0e-12)
    np.testing.assert_allclose(
        actual.parallel_dv01,
        (parallel_down - parallel_up) / (2.0 * bump) * 1.0e-4,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        actual.key_rate_duration,
        actual.key_rate_dv01 / (base[:, None] * 1.0e-4),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


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


@pytest.mark.parametrize(
    ("bond", "curve", "settlement", "target"),
    [
        (
            qfin.FixedRateBond(17.35, 0.071, face_value=1_000.0, frequency=4),
            qfin.YieldCurve(
                [0.0, 0.5, 3.0, 12.0, 30.0],
                [-0.01, 0.005, 0.04, 0.065, 0.025],
                interpolation="monotone_zero",
                extrapolation="flat_forward",
            ),
            0.37,
            975.25,
        ),
        (
            qfin.FixedRateBond.from_dates(
                "2023-11-30",
                "2044-02-29",
                0.02,
                face_value=250.0,
                frequency=2,
                end_of_month=True,
                first_coupon_date="2024-02-29",
            ),
            qfin.YieldCurve(
                [0.0, 2.0, 8.0, 25.0],
                [-0.015, 0.018, 0.055, 0.03],
                interpolation="log_linear_discount",
                extrapolation="flat_forward",
                valuation_date="2024-01-17",
                day_count="ACT/365 Fixed",
            ),
            "2024-01-17",
            247.75,
        ),
    ],
)
def test_par_yield_direct_schedule_matches_two_repricing_oracle(
    bond: qfin.FixedRateBond,
    curve: qfin.YieldCurve,
    settlement: Settlement,
    target: float,
) -> None:
    expected = _two_repricing_par_yield_oracle(
        bond,
        curve,
        settlement=settlement,
        target_clean_price=target,
    )
    actual = qfin.par_yield(
        bond,
        curve,
        settlement=settlement,
        target_clean_price=target,
    )
    assert actual == pytest.approx(expected, rel=2.0e-14, abs=2.0e-14)
    repriced = qfin.price_bonds(
        _with_coupon_rate(bond, actual),
        curve,
        settlement=settlement,
        engine="numpy",
    )
    assert repriced.clean_prices[0] == pytest.approx(target, rel=2.0e-14, abs=2.0e-11)


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
