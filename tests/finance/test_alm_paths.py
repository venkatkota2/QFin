import numpy as np
import pytest

import qfin


def _model() -> qfin.ALMModel:
    curve = qfin.YieldCurve([0, 1, 5, 10], [0.02, 0.025, 0.03, 0.035])
    assets = qfin.AssetPortfolio(
        [qfin.FixedRateBond(4, 0.03), qfin.FixedRateBond(8, 0.04)],
        [10, 8],
        equity_value=500,
        cash_value=50,
    )
    liabilities = qfin.LiabilityPortfolio.from_arrays(
        [2, 5, 9],
        [500, 700, 900],
        inflation_linkage=[1, 1, 1],
    )
    return qfin.ALMModel(assets, liabilities, curve)


def _scenarios(model: qfin.ALMModel) -> qfin.EconomicScenarioSet:
    rates = np.zeros((6, 3, model.curve.times.size))
    rates[:] = np.linspace(-0.005, 0.01, 6)[:, None, None]
    return qfin.EconomicScenarioSet(
        rates,
        credit_spread_shocks=np.linspace(0, 0.003, 6)[:, None] * np.ones((6, 3)),
        equity_returns=np.linspace(-0.10, 0.10, 6)[:, None] * np.ones((6, 3)),
        inflation_rates=np.linspace(0.0, 0.03, 6)[:, None] * np.ones((6, 3)),
        probabilities=np.arange(1, 7),
    )


def test_one_period_factor_attribution_reconciles_exactly() -> None:
    model = _model()
    scenarios = _scenarios(model)
    result = model.run_factor_scenarios(scenarios, engine="numpy", chunk_size=2)
    np.testing.assert_allclose(
        result.attribution.total_change,
        result.attribution.impacts.sum(axis=1) + result.attribution.interaction,
        atol=1e-12,
    )
    assert result.attribution.factor_names == (
        "rates",
        "credit_spread",
        "equity",
        "inflation",
    )
    assert result.loss_distribution().probabilities.tolist() == pytest.approx(
        scenarios.probabilities
    )
    sensitivity = model.sensitivities(engine="native")
    assert sensitivity.rate_impact != 0
    assert sensitivity.credit_spread_impact < 0
    assert sensitivity.equity_impact > 0
    assert sensitivity.inflation_impact < 0


@pytest.mark.parametrize(
    "strategy",
    [
        qfin.RebalancingStrategy(),
        qfin.RebalancingStrategy(
            target_equity_weight=0.30,
            rebalance_frequency=1,
            transaction_cost_rate=0.001,
        ),
    ],
)
def test_native_path_projection_matches_numpy_and_is_chunk_invariant(
    strategy: qfin.RebalancingStrategy,
) -> None:
    model = _model()
    scenarios = _scenarios(model)
    reference = model.project_paths(
        scenarios, strategy=strategy, engine="numpy", scenario_chunk_size=2
    )
    native = model.project_paths(
        scenarios, strategy=strategy, engine="native", scenario_chunk_size=4
    )
    for name in (
        "asset_values",
        "bond_values",
        "cash_values",
        "equity_values",
        "liability_values",
        "liability_payments",
        "surplus",
        "funding_ratio",
        "transaction_costs",
    ):
        np.testing.assert_allclose(
            getattr(native, name), getattr(reference, name), rtol=1e-13, atol=1e-10
        )
    assert native.asset_values.shape == (6, 4)
    assert np.all(native.cash_values >= 0)
    assert native.loss_distribution().losses.shape == (6,)
    if strategy.target_equity_weight is not None:
        assert np.any(native.transaction_costs > 0)


def test_auto_path_dispatch_follows_measured_numpy_policy() -> None:
    result = _model().project_paths(_scenarios(_model()))
    assert result.engine == "numpy"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_equity_weight": -0.01},
        {"target_equity_weight": 1.01},
        {"rebalance_frequency": 0},
        {"transaction_cost_rate": 1.0},
    ],
)
def test_rebalancing_strategy_rejects_invalid_controls(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        qfin.RebalancingStrategy(**kwargs)  # type: ignore[arg-type]
