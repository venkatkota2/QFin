from math import exp

import numpy as np
import pytest

import qfin


def test_curve_interpolation_discounting_and_forward_rate() -> None:
    curve = qfin.YieldCurve(
        times=np.array([0.0, 1.0, 3.0]),
        zero_rates=np.array([0.02, 0.03, 0.05]),
    )
    assert curve.zero_rate(2.0) == pytest.approx(0.04)
    assert curve.discount(2.0) == pytest.approx(exp(-0.08))
    assert curve.forward_rate(1.0, 3.0) == pytest.approx(0.06)
    np.testing.assert_allclose(curve.zero_rate([-1 + 1, 4]), [0.02, 0.05])


def test_curve_supports_negative_rates_and_node_shocks() -> None:
    curve = qfin.YieldCurve([0.0, 2.0, 10.0], [-0.01, 0.0, 0.02])
    assert curve.discount(0.5) > 1.0
    shifted = curve.shifted(np.array([0.01, 0.0, -0.01]))
    np.testing.assert_allclose(shifted.zero_rates, [0.0, 0.0, 0.01])
    parallel = curve.shifted(0.005)
    np.testing.assert_allclose(parallel.zero_rates, curve.zero_rates + 0.005)


def test_curve_accepts_explicit_compounding_without_losing_metadata() -> None:
    curve = qfin.YieldCurve(
        [0.0, 1.0, 5.0],
        [0.04, 0.04, 0.04],
        compounding="annual",
        valuation_date="2026-01-01",
        day_count="ACT/365 Fixed",
    )
    assert curve.quote_compounding is qfin.Compounding.ANNUAL
    assert curve.discount(5.0) == pytest.approx(1 / 1.04**5)
    assert curve.quoted_zero_rate(3.0) == pytest.approx(0.04)
    assert curve.quoted_zero_rate(0.0) == pytest.approx(0.04)
    assert curve.discount_date("2027-01-01") == pytest.approx(1 / 1.04)
    assert curve.explain()["canonical_rate_compounding"] == "continuous"


@pytest.mark.parametrize(
    "interpolation",
    ["linear_zero", "linear_discount", "log_linear_discount", "monotone_zero"],
)
def test_curve_interpolation_methods_reproduce_nodes(interpolation: str) -> None:
    times = np.array([0.0, 1.0, 3.0, 8.0])
    discounts = np.array([1.0, 0.98, 0.91, 0.72])
    curve = qfin.YieldCurve.from_discount_factors(
        times,
        discounts,
        interpolation=interpolation,
    )
    np.testing.assert_allclose(curve.discount(times), discounts, rtol=1e-14, atol=1e-14)


def test_discount_and_forward_rate_curve_constructors() -> None:
    discount_curve = qfin.YieldCurve.from_discount_factors(
        [0.0, 1.0, 2.0], [1.0, 0.97, 0.93]
    )
    np.testing.assert_allclose(discount_curve.node_discount_factors, [1.0, 0.97, 0.93])

    forward_curve = qfin.YieldCurve.from_forward_rates(
        [0.0, 1.0, 2.0], [0.02, 0.03], compounding="annual"
    )
    assert forward_curve.discount(1.0) == pytest.approx(1 / 1.02)
    assert forward_curve.discount(2.0) == pytest.approx(1 / (1.02 * 1.03))
    assert forward_curve.input_type == "forward_rate"


def test_market_quote_metadata_diagnostics_and_extrapolation() -> None:
    quotes = [
        qfin.CurveMarketQuote(0.0, 0.01, identifier="overnight"),
        qfin.CurveMarketQuote(1.0, -0.01, identifier="one-year"),
        qfin.CurveMarketQuote(2.0, 0.02, identifier="two-year"),
    ]
    curve = qfin.YieldCurve.from_market_quotes(quotes)
    assert curve.market_quotes == tuple(quotes)
    diagnostics = curve.diagnostics()
    assert diagnostics.negative_forward_intervals == (0,)
    assert diagnostics.warnings

    bounded = qfin.YieldCurve([1.0, 2.0], [0.01, 0.02], extrapolation="error")
    with pytest.raises(ValueError, match="outside"):
        bounded.discount(0.5)

    flat_forward = qfin.YieldCurve(
        [1.0, 2.0], [0.01, 0.02], extrapolation="flat_forward"
    )
    assert flat_forward.discount(0.0) == 1.0


def test_advanced_curve_interpolation_forces_accuracy_reference_path() -> None:
    curve = qfin.YieldCurve.from_discount_factors(
        [0.0, 1.0, 2.0],
        [1.0, 0.97, 0.92],
        interpolation="log_linear_discount",
    )
    bond = qfin.FixedRateBond(2.0, 0.0)
    result = qfin.price_bonds(bond, curve, engine="auto")
    assert result.engine == "numpy"
    assert result.dirty_prices[0] == pytest.approx(92.0)
    with pytest.raises(ValueError, match="native engine requires"):
        qfin.price_bonds(bond, curve, engine="native")


@pytest.mark.parametrize(
    "times,rates,message",
    [
        ([0.0, 1.0], [0.02], "equal"),
        ([0.0, 0.0], [0.02, 0.03], "increasing"),
        ([-1.0, 1.0], [0.02, 0.03], "non-negative"),
        ([0.0, 1.0], [0.02, float("nan")], "finite"),
    ],
)
def test_curve_rejects_malformed_inputs(
    times: list[float], rates: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        qfin.YieldCurve(times, rates)
