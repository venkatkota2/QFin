import numpy as np
import pytest

import qfin


def _alm_model() -> qfin.ALMModel:
    curve = qfin.YieldCurve([0.0, 1.0, 5.0, 10.0], [0.02, 0.025, 0.03, 0.035])
    assets = qfin.AssetPortfolio(
        [
            qfin.FixedRateBond(3.0, 0.03),
            qfin.FixedRateBond(8.0, 0.04),
        ],
        [8.0, 12.0],
    )
    liabilities = qfin.LiabilityPortfolio.from_arrays(
        [2.0, 5.0, 9.0], [500.0, 700.0, 900.0]
    )
    return qfin.ALMModel(assets, liabilities, curve)


def test_alm_evaluation_fields_and_gap_definition() -> None:
    result = _alm_model().evaluate(engine="numpy")
    assert result.surplus == pytest.approx(result.asset_pv - result.liability_pv)
    assert result.funding_ratio == pytest.approx(result.asset_pv / result.liability_pv)
    assert result.duration_gap == pytest.approx(
        result.asset_duration
        - (result.liability_pv / result.asset_pv) * result.liability_duration
    )
    assert result.asset_convexity > 0
    assert result.liability_convexity > 0


def test_parallel_rate_scenarios_reprice_both_sides() -> None:
    model = _alm_model()
    scenarios = qfin.RateScenarioSet.parallel(model.curve, [-0.01, 0.0, 0.01])
    result = model.run_scenarios(scenarios, engine="numpy", chunk_size=2)
    assert result.labels[1].startswith("parallel_1_")
    assert result.asset_pv[0] > result.asset_pv[1] > result.asset_pv[2]
    assert result.liability_pv[0] > result.liability_pv[1] > result.liability_pv[2]
    assert result.asset_pv[1] == pytest.approx(model.evaluate(engine="numpy").asset_pv)
    assert result.loss_distribution().losses.shape == (3,)


def test_steepener_and_key_rate_scenario_shapes() -> None:
    curve = _alm_model().curve
    steepener = qfin.RateScenarioSet.steepener(
        curve, short_shift=-0.005, long_shift=0.01
    )
    key_rate = qfin.RateScenarioSet.key_rate(
        curve, key_time=5.0, shift=0.01, width=4.0
    )
    assert steepener.shocks.shape == key_rate.shocks.shape == (1, curve.times.size)
    assert key_rate.shocks[0, 2] == pytest.approx(0.01)


def test_single_node_curve_scenarios_are_supported() -> None:
    curve = qfin.YieldCurve([0.0], [0.03])
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(1.0, 0.0)], [1.0]),
        qfin.LiabilityPortfolio.from_arrays([1.0], [95.0]),
        curve,
    )
    scenarios = qfin.RateScenarioSet.parallel(curve, [0.0, 0.01])
    result = model.run_scenarios(scenarios, engine="numpy")
    assert np.all(np.isfinite(result.surplus))
