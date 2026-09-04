from math import exp

import numpy as np
import pytest

import qfin


def test_mixed_instrument_bootstrap_recovers_flat_curve_and_residuals() -> None:
    flat_rate = 0.04
    reference = qfin.YieldCurve(
        [0.0, 5.0],
        [flat_rate, flat_rate],
        interpolation="log_linear_discount",
    )
    deposit_rate = (exp(flat_rate * 0.5) - 1.0) / 0.5
    bond = qfin.FixedRateBond(2.0, 0.05, frequency=2)
    bond_price = float(qfin.price_bonds(bond, reference, engine="numpy").clean_prices[0])
    swap_times = np.arange(0.5, 3.01, 0.5)
    swap_discounts = np.exp(-flat_rate * swap_times)
    swap_rate = (1.0 - swap_discounts[-1]) / (0.5 * np.sum(swap_discounts))

    report = qfin.bootstrap_curve(
        [
            qfin.Deposit(0.5, deposit_rate, identifier="6m-deposit"),
            qfin.ZeroCouponInstrument(1.0, 100.0 * exp(-flat_rate), identifier="1y-zero"),
            qfin.BondMarketQuote(bond, bond_price, identifier="2y-bond"),
            qfin.SimpleSwap(3.0, swap_rate, identifier="3y-swap"),
        ],
        tolerance=1.0e-10,
    )
    assert report.success
    assert report.curve.input_type == "bootstrapped_instruments"
    assert report.maximum_absolute_residual <= report.tolerance
    np.testing.assert_allclose(report.discount_factors, np.exp(-flat_rate * report.node_times))
    np.testing.assert_allclose(report.forward_rates, flat_rate, atol=1.0e-12)
    assert [item.instrument_type for item in report.instruments] == [
        "deposit",
        "zero_coupon",
        "bond",
        "swap",
    ]


def test_dated_bootstrap_reprices_deposit_zero_and_bond() -> None:
    valuation_date = "2024-01-15"
    flat_rate = 0.03
    reference = qfin.YieldCurve(
        [0.0, 3.0],
        [flat_rate, flat_rate],
        interpolation="log_linear_discount",
        valuation_date=valuation_date,
    )
    deposit_maturity = "2024-07-15"
    curve_time = qfin.year_fraction(valuation_date, deposit_maturity, "ACT/365 Fixed")
    deposit_accrual = qfin.year_fraction(valuation_date, deposit_maturity, "ACT/360")
    deposit_rate = (exp(flat_rate * curve_time) - 1.0) / deposit_accrual
    zero_maturity = "2025-01-15"
    zero_price = 100.0 * float(reference.discount_date(zero_maturity))
    bond = qfin.FixedRateBond.from_dates(
        valuation_date,
        "2026-01-15",
        0.04,
        frequency=2,
    )
    bond_price = float(qfin.price_bonds(bond, reference, engine="numpy").clean_prices[0])

    report = qfin.bootstrap_curve(
        [
            qfin.Deposit(deposit_maturity, deposit_rate),
            qfin.ZeroCouponInstrument(zero_maturity, zero_price),
            qfin.BondMarketQuote(bond, bond_price, identifier="dated-bond"),
        ],
        valuation_date=valuation_date,
    )
    assert report.success
    assert report.curve.valuation_date == qfin.as_date(valuation_date)
    assert report.maximum_absolute_residual < 1.0e-10


def test_bootstrap_rejects_duplicate_nodes_and_unbracketed_quotes() -> None:
    with pytest.raises(ValueError, match="unique"):
        qfin.bootstrap_curve([qfin.Deposit(1.0, 0.03), qfin.SimpleSwap(1.0, 0.04)])
    with pytest.raises(qfin.CurveBootstrapError, match="not bracketed"):
        qfin.bootstrap_curve([qfin.SimpleSwap(2.0, -10.0)])


def test_bootstrap_input_diagnostics() -> None:
    with pytest.raises(ValueError, match="at least one"):
        qfin.bootstrap_curve([])
    with pytest.raises(ValueError, match="valuation_date"):
        qfin.bootstrap_curve([qfin.Deposit("2027-01-01", 0.03)])
    with pytest.raises(ValueError, match="positive"):
        qfin.ZeroCouponInstrument(1.0, 0.0)

