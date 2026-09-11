"""Nonlinear curves, dated quotes, and financial-unit sensitivity references."""

import json
from datetime import date
from decimal import Decimal, localcontext
from math import exp
from pathlib import Path

import numpy as np
import pytest

import qfin

D = Decimal
DATA = Path(__file__).parent / "reference_data"


@pytest.mark.parametrize(
    "case", json.loads((DATA / "dated_bonds.json").read_text()), ids=lambda case: case["id"]
)
@pytest.mark.parametrize("engine", ["numpy", "native"])
def test_quantlib_dated_bond_money(case, engine):
    if engine == "native" and not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    bond = qfin.FixedRateBond.from_dates(
        case["issue"],
        case["maturity"],
        case["coupon"],
        face_value=case["face"],
        day_count=case["coupon_day_count"],
        business_day_convention="unadjusted",
        end_of_month=True,
    )
    curve = qfin.YieldCurve([0, 100], [case["rate"]] * 2, valuation_date=case["settlement"])
    actual = qfin.price_bonds(bond, curve, engine=engine)
    for attribute, key in [
        ("dirty_prices", "dirty_price"),
        ("clean_prices", "clean_price"),
        ("accrued_interest", "accrued_interest"),
    ]:
        difference = abs(getattr(actual, attribute)[0] - case[key])
        assert difference <= max(1e-10, case["face"] * 2e-12)
        assert getattr(actual, attribute)[0] == pytest.approx(case[key], rel=2e-12, abs=1e-10)


@pytest.mark.parametrize("rate", [-0.01, 1e-12, 0.08])
@pytest.mark.parametrize("face", [1e-6, 100, 1e12])
def test_dated_bootstrap_from_independent_calendar_dcf(rate, face):
    valuation = date(2024, 1, 15)
    deposit_end, zero_end, bond_end = date(2024, 7, 15), date(2025, 1, 15), date(2026, 1, 15)

    def df(end):
        return exp(-rate * (end - valuation).days / 365)

    deposit_rate = (1 / df(deposit_end) - 1) / ((deposit_end - valuation).days / 360)
    # Exact regular 30E/360 coupons; discount clock is independently ACT/365F.
    payments = [date(2024, 7, 15), date(2025, 1, 15), date(2025, 7, 15), bond_end]
    price = sum(0.02 * face * df(end) for end in payments) + face * df(bond_end)
    bond = qfin.FixedRateBond.from_dates(
        valuation,
        bond_end,
        0.04,
        face_value=face,
        day_count="30E/360",
        business_day_convention="unadjusted",
    )
    result = qfin.bootstrap_curve(
        [
            qfin.Deposit(deposit_end, deposit_rate),
            qfin.ZeroCouponInstrument(zero_end, 100 * df(zero_end)),
            qfin.BondMarketQuote(bond, price),
        ],
        valuation_date=valuation,
        interpolation="log_linear_discount",
        tolerance=max(1e-10, 4 * float(np.spacing(price))),
    )
    assert result.success and all(item.passed for item in result.instruments)
    np.testing.assert_allclose(
        result.discount_factors, [1, df(deposit_end), df(zero_end), df(bond_end)], rtol=2e-10
    )
    assert abs(qfin.price_bonds(bond, result.curve).clean_prices[0] - price) <= max(
        1e-9, face * 1e-10
    )


@pytest.mark.parametrize(
    "method", ["linear_zero", "linear_discount", "log_linear_discount", "monotone_zero"]
)
@pytest.mark.parametrize("query", [2, 5])
def test_nonlinear_curve_independent_hermite_and_discount_formulas(method, query):
    # Three-node turning point. PCHIP slopes from the shape-preserving endpoint
    # formula are 11/300, 0, -3/100 (last endpoint limited to 3*secant).
    with localcontext() as context:
        context.prec = 70
        times, rates, slopes = (
            list(map(D, [1, 3, 7])),
            [D(".01"), D(".06"), D(".02")],
            [D(11) / 300, D(0), D("-.03")],
        )
        i = 0 if query == 2 else 1
        h = times[i + 1] - times[i]
        x = (D(query) - times[i]) / h
        left, right = (-times[i] * rates[i]).exp(), (-times[i + 1] * rates[i + 1]).exp()
        if method == "linear_discount":
            expected = left * (1 - x) + right * x
        elif method == "log_linear_discount":
            expected = ((1 - x) * left.ln() + x * right.ln()).exp()
        else:
            rate = rates[i] * (1 - x) + rates[i + 1] * x
            if method == "monotone_zero":
                rate = (
                    (2 * x**3 - 3 * x**2 + 1) * rates[i]
                    + (x**3 - 2 * x**2 + x) * h * slopes[i]
                    + (-2 * x**3 + 3 * x**2) * rates[i + 1]
                    + (x**3 - x**2) * h * slopes[i + 1]
                )
            expected = (-rate * D(query)).exp()
    curve = qfin.YieldCurve([1, 3, 7], [0.01, 0.06, 0.02], interpolation=method)
    assert curve.discount(query) == pytest.approx(float(expected), rel=3e-14)


@pytest.mark.parametrize("time", [1e-6, 0.5, 60])
@pytest.mark.parametrize("rate", [-0.02, 1e-12, 0.1])
def test_zero_coupon_sensitivities_against_decimal(time, rate):
    with localcontext() as context:
        context.prec = 70
        t, r, bump, face = D(str(time)), D(str(rate)), D(".0001"), D("1e12")
        price = face * (-r * t).exp()
        up, down = face * (-(r + bump) * t).exp(), face * (-(r - bump) * t).exp()
        duration = (down - up) / (2 * bump * price)
        convexity = (down + up - 2 * price) / (bump * bump * price)
    bond = qfin.FixedRateBond(time, 0, face_value=float(face))
    curve = qfin.YieldCurve([0, 100], [rate, rate])
    result = qfin.price_bonds(bond, curve, engine="numpy")
    assert result.parallel_zero_duration[0] == pytest.approx(time, rel=3e-13)
    assert result.spread_duration[0] == pytest.approx(time, rel=3e-13)
    assert result.cs01[0] == pytest.approx(float((down - up) / 2), rel=3e-12, abs=1e-8)
    assert result.effective_duration[0] == pytest.approx(float(duration), rel=3e-12)
    assert result.effective_convexity[0] == pytest.approx(float(convexity), rel=3e-12)
