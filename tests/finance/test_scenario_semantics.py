from __future__ import annotations

import numpy as np
import pytest

import qfin
from qfin.finance import scenarios as scenario_module
from qfin.finance.scenarios import (
    EconomicScenarioSet,
    scenario_indexed_cashflow_values,
    scenario_portfolio_values,
)

INTERPOLATIONS = tuple(item.value for item in qfin.CurveInterpolation)
EXTRAPOLATIONS = tuple(item.value for item in qfin.CurveExtrapolation)


def test_automatic_memory_bound_preserves_scenario_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = qfin.YieldCurve([0, 5, 50], [-0.01, 0.04, 0.02])
    times = np.linspace(0.1, 50, 101)
    amounts = np.linspace(-1e6, 2e6, times.size)
    offsets = np.array([0, 30, 30, 101], dtype=np.int64)
    weights = np.array([1.0, 0.0, 0.5])
    scenarios = qfin.RateScenarioSet(np.random.default_rng(63).normal(0, 0.01, (27, 3)))
    reference, _ = scenario_portfolio_values(
        times, amounts, offsets, weights, curve, scenarios, engine="numpy", chunk_size=1,
    )
    observed_chunk_sizes: list[int] = []
    original = scenario_module._PreparedScenarioValuation.discount_factors

    def observe(self: object, shocks: np.ndarray) -> np.ndarray:
        observed_chunk_sizes.append(shocks.shape[0])
        return original(self, shocks)

    monkeypatch.setattr(scenario_module, "_MAX_SCENARIO_MATRIX_ELEMENTS", 303)
    monkeypatch.setattr(scenario_module._PreparedScenarioValuation, "discount_factors", observe)
    result, _ = scenario_portfolio_values(
        times, amounts, offsets, weights, curve, scenarios, engine="numpy", chunk_size=100_000,
    )
    np.testing.assert_array_equal(result, reference)
    assert max(observed_chunk_sizes) <= 3
    assert len(observed_chunk_sizes) == 9


def _scenario_shocks(node_count: int) -> np.ndarray:
    rng = np.random.default_rng(20260904)
    random_shocks = rng.uniform(-0.025, 0.035, size=(7, node_count))
    nonparallel = np.linspace(-0.02, 0.03, node_count)
    return np.vstack(
        (
            np.zeros((1, node_count)),
            np.full((1, node_count), 0.0125),
            np.full((1, node_count), -0.0075),
            nonparallel[None, :],
            random_shocks,
        )
    )


def _slow_portfolio_oracle(
    curve: qfin.YieldCurve,
    shocks: np.ndarray,
    times: np.ndarray,
    amounts: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    result = np.empty(shocks.shape[0], dtype=np.float64)
    for scenario_index, shock in enumerate(shocks):
        stressed_curve = curve.shifted(shock)
        value = 0.0
        for position, weight in enumerate(weights):
            start = offsets[position]
            stop = offsets[position + 1]
            value += weight * float(
                np.sum(
                    amounts[start:stop] * stressed_curve.discount(times[start:stop]),
                    dtype=np.float64,
                )
            )
        result[scenario_index] = value
    return result


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("extrapolation", EXTRAPOLATIONS)
@pytest.mark.parametrize(
    "rates",
    [
        (0.01, 0.025, 0.04, 0.055),
        (0.0, 0.0, 0.0, 0.0),
        (-0.03, -0.02, -0.01, -0.005),
        (-0.01, 0.0, 0.05, 0.12),
        (0.08, 0.06, 0.03, 0.01),
    ],
    ids=("positive", "zero", "negative", "steep", "inverted"),
)
def test_fast_scenarios_match_shifted_curve_oracle(
    interpolation: str,
    extrapolation: str,
    rates: tuple[float, ...],
) -> None:
    curve = qfin.YieldCurve(
        [0.25, 1.0, 5.0, 30.0],
        rates,
        interpolation=interpolation,
        extrapolation=extrapolation,
    )
    if extrapolation == "error":
        times = np.array([0.25, 0.4, 1.0, 2.7, 5.0, 12.0, 30.0])
        offsets = np.array([0, 2, 2, 5, 7], dtype=np.int64)
    else:
        times = np.array([0.01, 0.25, 0.4, 1.0, 2.7, 5.0, 12.0, 30.0, 50.0])
        offsets = np.array([0, 3, 3, 7, 9], dtype=np.int64)
    amounts = np.linspace(1.0e-3, 8.0e5, times.size)
    amounts[1::3] *= -0.2
    weights = np.array([1.5, -0.25, 0.0, 2.25])
    shocks = _scenario_shocks(curve.times.size)
    scenarios = qfin.RateScenarioSet(shocks)

    actual, engine = scenario_portfolio_values(
        times,
        amounts,
        offsets,
        weights,
        curve,
        scenarios,
        engine="numpy",
        chunk_size=3,
    )
    expected = _slow_portfolio_oracle(curve, shocks, times, amounts, offsets, weights)

    assert engine == "numpy"
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-9)


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("extrapolation", EXTRAPOLATIONS)
def test_single_node_scenarios_match_shifted_curve(
    interpolation: str,
    extrapolation: str,
) -> None:
    curve = qfin.YieldCurve(
        [2.0],
        [-0.015],
        interpolation=interpolation,
        extrapolation=extrapolation,
    )
    times = np.array([2.0]) if extrapolation == "error" else np.array([0.0, 2.0, 75.0])
    amounts = np.array([100.0]) if extrapolation == "error" else np.array([1.0, 2.0, 3.0])
    offsets = np.array([0, times.size], dtype=np.int64)
    weights = np.array([4.0])
    shocks = np.array([[0.0], [0.01], [-0.02]])

    actual, _ = scenario_portfolio_values(
        times,
        amounts,
        offsets,
        weights,
        curve,
        qfin.RateScenarioSet(shocks),
        engine="numpy",
        chunk_size=1,
    )
    expected = _slow_portfolio_oracle(curve, shocks, times, amounts, offsets, weights)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=1.0e-12)


def test_scenario_permutation_and_chunk_size_only_permute_results() -> None:
    curve = qfin.YieldCurve(
        [0.0, 1.0, 7.0, 40.0],
        [-0.01, 0.02, 0.055, 0.025],
        interpolation="monotone_zero",
        extrapolation="flat_forward",
    )
    shocks = _scenario_shocks(curve.times.size)
    permutation = np.array([7, 0, 10, 3, 1, 5, 9, 2, 8, 4, 6])
    times = np.array([0.0, 0.01, 1.0, 3.0, 7.0, 19.0, 40.0, 80.0])
    amounts = np.array([1.0, -3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0])
    offsets = np.array([0, 1, 4, 8], dtype=np.int64)
    weights = np.array([2.0, -1.0, 0.5])

    baseline, _ = scenario_portfolio_values(
        times,
        amounts,
        offsets,
        weights,
        curve,
        qfin.RateScenarioSet(shocks),
        engine="numpy",
        chunk_size=1,
    )
    permuted, _ = scenario_portfolio_values(
        times,
        amounts,
        offsets,
        weights,
        curve,
        qfin.RateScenarioSet(shocks[permutation]),
        engine="numpy",
        chunk_size=7,
    )
    np.testing.assert_array_equal(permuted, baseline[permutation])


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("extrapolation", ("flat_zero", "flat_forward"))
def test_indexed_scenarios_use_shifted_curve_semantics(
    interpolation: str,
    extrapolation: str,
) -> None:
    curve = qfin.YieldCurve(
        [0.5, 2.0, 10.0, 35.0],
        [-0.02, 0.01, 0.07, 0.03],
        interpolation=interpolation,
        extrapolation=extrapolation,
    )
    shocks = _scenario_shocks(curve.times.size)
    inflation = np.linspace(-0.01, 0.08, shocks.shape[0])
    scenarios = EconomicScenarioSet(
        shocks[:, None, :],
        inflation_rates=inflation[:, None],
    )
    times = np.array([0.01, 0.5, 1.25, 10.0, 20.0, 60.0])
    amounts = np.array([2.0, -5.0, 7.0, 11.0, 13.0, 17.0])
    linkages = np.array([0.0, 1.0, 0.5, 1.0, 0.25, 2.0])

    actual, _ = scenario_indexed_cashflow_values(
        times,
        amounts,
        linkages,
        curve,
        scenarios,
        engine="numpy",
        chunk_size=4,
    )
    expected = np.array(
        [
            np.sum(
                amounts
                * np.power(1.0 + inflation[index], times * linkages)
                * curve.shifted(shock).discount(times),
                dtype=np.float64,
            )
            for index, shock in enumerate(shocks)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=1.0e-12)


def test_native_scenario_kernel_matches_exact_supported_numpy_semantics() -> None:
    if not qfin.system_info()["native_extension"]:
        pytest.skip("native extension unavailable")
    rng = np.random.default_rng(19)
    curve = qfin.YieldCurve(
        [0.0, 0.5, 3.0, 15.0, 60.0],
        [-0.01, 0.0, 0.025, 0.06, 0.035],
        interpolation="linear_zero",
        extrapolation="flat_zero",
    )
    times = rng.uniform(0.0, 75.0, size=173)
    amounts = rng.uniform(-1.0e6, 1.0e6, size=times.size)
    offsets = np.array([0, 1, 1, 20, 90, 173], dtype=np.int64)
    weights = rng.uniform(-2.0, 2.0, size=offsets.size - 1)
    scenarios = qfin.RateScenarioSet(rng.uniform(-0.03, 0.04, size=(37, 5)))

    reference, _ = scenario_portfolio_values(
        times, amounts, offsets, weights, curve, scenarios, engine="numpy", chunk_size=13
    )
    native, selected = scenario_portfolio_values(
        times, amounts, offsets, weights, curve, scenarios, engine="native", chunk_size=11
    )

    assert selected == "native"
    np.testing.assert_allclose(native, reference, rtol=2.0e-13, atol=2.0e-8)


def test_error_extrapolation_rejects_out_of_domain_cashflows() -> None:
    curve = qfin.YieldCurve(
        [1.0, 5.0],
        [0.02, 0.03],
        interpolation="linear_discount",
        extrapolation="error",
    )
    with pytest.raises(ValueError, match="outside the curve domain"):
        scenario_portfolio_values(
            np.array([0.5, 2.0]),
            np.array([1.0, 1.0]),
            np.array([0, 2]),
            np.array([1.0]),
            curve,
            qfin.RateScenarioSet([[0.0, 0.0]]),
            engine="numpy",
        )
