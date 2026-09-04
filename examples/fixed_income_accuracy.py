"""Dated bond, bootstrap, risk, and independent-validation workflow."""

from math import exp

import qfin

valuation_date = "2026-01-30"
bond = qfin.FixedRateBond.from_dates(
    "2025-01-31",
    "2031-01-31",
    0.04,
    frequency=2,
    day_count="30/360",
    end_of_month=True,
)
curve = qfin.YieldCurve(
    [0.0, 1.0, 5.0, 10.0],
    [0.03, 0.032, 0.035, 0.038],
    interpolation="log_linear_discount",
    valuation_date=valuation_date,
)
analytics = qfin.price_bonds(bond, curve, z_spread=0.005, engine="numpy")
keys = qfin.key_rate_risk(bond, curve, engine="numpy")
coupon_at_par = qfin.par_yield(bond, curve)

flat_rate = 0.035
deposit_rate = (exp(flat_rate * 0.5) - 1.0) / 0.5
bootstrap = qfin.bootstrap_curve(
    [
        qfin.Deposit(0.5, deposit_rate, identifier="6m-deposit"),
        qfin.ZeroCouponInstrument(1.0, 100.0 * exp(-flat_rate), identifier="1y-zero"),
        qfin.SimpleSwap(
            2.0,
            (1.0 - exp(-flat_rate * 2.0))
            / (0.5 * sum(exp(-flat_rate * time) for time in (0.5, 1.0, 1.5, 2.0))),
            identifier="2y-swap",
        ),
    ]
)

reference = qfin.reference_bond_from_yield(
    bond,
    0.04,
    settlement=valuation_date,
)
validation = qfin.validate_financial_values(
    "dated bond clean price",
    [qfin.price_bonds_from_yield(bond, 0.04, settlement=valuation_date).clean_prices[0]],
    [reference.clean_price],
    tolerance=qfin.FinancialTolerance(financial=1.0e-8, unit="price units"),
)
validation.assert_valid()

print(
    {
        "clean_price": float(analytics.clean_prices[0]),
        "accrued_interest": float(analytics.accrued_interest[0]),
        "parallel_zero_duration": float(analytics.parallel_zero_duration[0]),
        "cs01": float(analytics.cs01[0]),
        "par_yield": coupon_at_par,
        "key_rate_dv01": keys.key_rate_dv01[0].tolist(),
        "bootstrap": bootstrap.explain(),
        "validation": validation.explain(),
    }
)
