import numpy as np
import pytest

import qfin


def test_discount_curve_interpolation_and_shift() -> None:
    curve = qfin.DiscountCurve(
        times=np.array([0.0, 5.0, 10.0]),
        zero_rates=np.array([0.02, 0.03, 0.04]),
    )
    assert curve.zero_rate([2.5, 7.5]).tolist() == pytest.approx([0.025, 0.035])
    assert curve.discount_factor(5.0) == pytest.approx(np.exp(-0.15))
    assert curve.shifted(0.01).zero_rate([5.0])[0] == pytest.approx(0.04)


def test_cash_flow_schedule_aggregates_dates_and_sensitivities() -> None:
    schedule = qfin.CashFlowSchedule(
        times=np.array([2.0, 1.0, 2.0]),
        amounts=np.array([10.0, 5.0, 15.0]),
    )
    assert schedule.times.tolist() == [1.0, 2.0]
    assert schedule.amounts.tolist() == [5.0, 25.0]
    curve = qfin.DiscountCurve.flat(0.04)
    value = schedule.present_value(curve)
    epsilon = 1e-5
    numerical_duration = -(
        schedule.present_value(curve.shifted(epsilon))
        - schedule.present_value(curve.shifted(-epsilon))
    ) / (2 * epsilon * value)
    assert schedule.parallel_duration(curve) == pytest.approx(
        numerical_duration, rel=1e-8
    )
    assert schedule.parallel_convexity(curve) > 0


def test_bond_and_portfolio_cashflows_are_preaggregated() -> None:
    curve = qfin.DiscountCurve.flat(0.03)
    zero = qfin.FixedRateBond(
        face_value=1_000, coupon_rate=0.0, maturity=5.0, coupon_frequency=1
    )
    assert zero.price(curve) == pytest.approx(1_000 * np.exp(-0.15))
    coupon = qfin.FixedRateBond(
        face_value=1_000, coupon_rate=0.04, maturity=2.0, coupon_frequency=2
    )
    portfolio = qfin.FixedIncomePortfolio(
        (qfin.BondPosition(zero, 2.0), qfin.BondPosition(coupon, 3.0))
    )
    expected = 2.0 * zero.price(curve) + 3.0 * coupon.price(curve)
    assert portfolio.present_value(curve) == pytest.approx(expected)
    assert portfolio.cashflows.times.size == 5
    assert portfolio.duration(curve) > 0
    assert portfolio.convexity(curve) > 0


def test_invalid_bond_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="whole number"):
        qfin.FixedRateBond(1_000, 0.04, 1.1, coupon_frequency=2)
