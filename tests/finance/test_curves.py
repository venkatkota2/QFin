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


def test_diagnostics_inspect_monotone_zero_between_node_path() -> None:
    # Node discount factors decrease, but monotone interpolation of zero rates
    # still creates locally increasing discount factors in two intervals.
    curve = qfin.YieldCurve(
        [0.0, 0.25, 1.0, 3.0, 10.0],
        [
            0.5460085498532156,
            0.5460085498532156,
            0.1473609591218682,
            0.06383696589637107,
            0.022846682264777728,
        ],
        interpolation="monotone_zero",
        extrapolation="error",
    )

    diagnostics = curve.diagnostics(samples_per_interval=65)

    assert diagnostics.node_increasing_discount_intervals == ()
    assert diagnostics.between_node_increasing_discount_intervals == (1, 3)
    assert diagnostics.between_node_negative_forward_intervals == (1, 3)
    assert diagnostics.node_warnings == ()
    assert any("interpolation path" in warning for warning in diagnostics.interpolation_warnings)
    assert diagnostics.extrapolation_warnings == ()
    assert diagnostics.has_positive_discount_factors


def test_diagnostics_separate_extrapolation_and_extreme_forward_warnings() -> None:
    curve = qfin.YieldCurve(
        [1.0, 2.0],
        [-0.02, -0.04],
        interpolation="log_linear_discount",
        extrapolation="flat_zero",
    )

    diagnostics = curve.diagnostics(extreme_forward_rate=0.01)

    assert diagnostics.extrapolation_warning_sides == ("left", "right")
    assert diagnostics.extrapolation_warnings
    assert diagnostics.between_node_extreme_forward_intervals == (0,)
    assert diagnostics.minimum_forward_rate is not None
    assert diagnostics.minimum_forward_rate < 0.0


@pytest.mark.parametrize("invalid", [True, 2.5, np.float32(4.0)])
def test_diagnostics_reject_non_integer_sampling_counts(invalid: object) -> None:
    curve = qfin.YieldCurve([0.0, 1.0], [0.01, 0.02])
    with pytest.raises(ValueError, match="samples_per_interval must be an integer"):
        curve.diagnostics(samples_per_interval=invalid)  # type: ignore[arg-type]


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


def test_shifted_curve_preserves_and_accumulates_provenance() -> None:
    quotes = (
        qfin.CurveMarketQuote(1.0, 0.02, identifier="one-year"),
        qfin.CurveMarketQuote(5.0, 0.03, identifier="five-year"),
    )
    base = qfin.YieldCurve.from_market_quotes(
        quotes,
        compounding="annual",
        interpolation="monotone_zero",
        extrapolation="flat_forward",
        valuation_date="2026-01-02",
    )
    shocked = base.shifted(np.array([0.001, -0.002])).shifted(0.0005)

    explanation = shocked.explain()
    assert explanation["input_type"] == "shifted_zero_rate"
    assert explanation["origin_input_type"] == "market_quote"
    assert explanation["quote_compounding"] == "annual"
    assert explanation["interpolation"] == "monotone_zero"
    assert explanation["extrapolation"] == "flat_forward"
    assert explanation["valuation_date"] == "2026-01-02"
    assert explanation["applied_node_shock"] == pytest.approx([0.0015, -0.0015])
    assert explanation["originating_quote_metadata"] == [
        {
            "source": "direct_market_node",
            "identifier": "one-year",
            "quote_type": "zero_rate",
            "time": 1.0,
            "value": 0.02,
        },
        {
            "source": "direct_market_node",
            "identifier": "five-year",
            "quote_type": "zero_rate",
            "time": 5.0,
            "value": 0.03,
        },
    ]
    assert shocked.market_quotes == quotes
    np.testing.assert_allclose(
        shocked.zero_rates,
        base.zero_rates + np.array([0.0015, -0.0015]),
        rtol=0.0,
        atol=2.0e-17,
    )
