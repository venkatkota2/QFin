from math import fsum

import numpy as np
import pytest

import qfin
from qfin._numerics import stable_sum, stable_weighted_sum
from qfin.finance.fixed_income import _segment_sum
from qfin.finance.scenarios import RateScenarioSet, _segment_sum_rows, scenario_portfolio_values


def test_stable_sum_recovers_small_portfolio_residual_under_cancellation() -> None:
    notionals = np.concatenate(
        (
            np.full(100_000, 1.0e12),
            np.full(100_000, -1.0e12),
            np.ones(100_000),
        )
    )
    expected = fsum(float(value) for value in notionals)
    assert stable_sum(notionals) == expected


def test_stable_weighted_sum_matches_high_accuracy_financial_oracle() -> None:
    values = np.concatenate(
        (
            np.full(10_000, 1.0e12),
            np.full(10_000, -1.0e12),
            np.ones(10_000),
        )
    )
    weights = np.full(values.size, 1.0 / 3.0)
    expected = fsum(
        float(value) * float(weight)
        for value, weight in zip(values, weights, strict=True)
    )
    assert stable_weighted_sum(values, weights) == expected


def test_weighted_risk_moments_match_high_accuracy_oracle_across_engines() -> None:
    losses = np.concatenate(
        (
            np.full(4_000, 1.0e12),
            np.full(4_000, -1.0e12),
            np.ones(4_000),
        )
    )
    distribution = qfin.LossDistribution(losses)
    assert distribution.probabilities is not None
    expected = fsum(
        float(loss) * float(probability)
        for loss, probability in zip(
            distribution.losses, distribution.probabilities, strict=True
        )
    )
    numpy_result = qfin.aggregate_risk(distribution, confidence=0.95, engine="numpy")
    assert numpy_result.mean == expected
    if qfin.system_info()["native_extension"]:
        native_result = qfin.aggregate_risk(distribution, confidence=0.95, engine="native")
        assert native_result.mean == pytest.approx(expected, abs=2.0e-15)
        assert native_result.standard_deviation == pytest.approx(
            numpy_result.standard_deviation, rel=5.0e-16
        )


def test_segmented_reductions_support_empty_and_large_streams() -> None:
    rng = np.random.default_rng(1907)
    counts = np.asarray([0, 1, 4_997, 0, 9, 0, 5_003, 2, 0], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
    values = rng.normal(size=int(offsets[-1]))
    expected = np.asarray(
        [
            np.sum(values[offsets[index] : offsets[index + 1]], dtype=np.float64)
            for index in range(counts.size)
        ]
    )
    np.testing.assert_allclose(
        _segment_sum(values, offsets), expected, rtol=2.0e-15, atol=2.0e-14
    )

    rows = np.vstack((values, values[::-1], values * 1.0e9))
    row_expected = np.vstack(
        [
            [
                np.sum(row[offsets[index] : offsets[index + 1]], dtype=np.float64)
                for index in range(counts.size)
            ]
            for row in rows
        ]
    )
    np.testing.assert_allclose(
        _segment_sum_rows(rows, offsets), row_expected, rtol=2.0e-15, atol=2.0e-5
    )


def test_scenario_portfolio_uses_accurate_fallback_under_severe_cancellation() -> None:
    curve = qfin.YieldCurve([0.0], [0.0])
    amounts = np.asarray([1.0e12, 1.0, -1.0e12])
    weights = np.full(3, 1.0 / 3.0)
    offsets = np.arange(4, dtype=np.int64)
    scenarios = RateScenarioSet(np.zeros((2, 1)))
    expected = fsum(
        float(amount) * float(weight)
        for amount, weight in zip(amounts, weights, strict=True)
    )

    numpy_values, _ = scenario_portfolio_values(
        np.zeros(3), amounts, offsets, weights, curve, scenarios, engine="numpy"
    )
    np.testing.assert_array_equal(numpy_values, np.full(2, expected))
    if qfin.system_info()["native_extension"]:
        native_values, _ = scenario_portfolio_values(
            np.zeros(3), amounts, offsets, weights, curve, scenarios, engine="native"
        )
        np.testing.assert_array_equal(native_values, np.full(2, expected))
