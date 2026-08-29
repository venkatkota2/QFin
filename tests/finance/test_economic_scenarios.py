import numpy as np
import pytest

import qfin


def _curve() -> qfin.YieldCurve:
    return qfin.YieldCurve([0, 1, 5], [0.02, 0.025, 0.03])


def test_economic_scenario_paths_validate_and_normalize_probabilities() -> None:
    source_rates = np.zeros((2, 3, 3))
    source_equity = np.array([[0.1, 0.0, -0.1], [0.0, 0.1, 0.2]])
    scenarios = qfin.EconomicScenarioSet(
        source_rates,
        equity_returns=source_equity,
        inflation_rates=0.02,
        probabilities=[2, 1],
        labels=("up", "down"),
    )
    assert scenarios.scenario_count == 2
    assert scenarios.period_count == 3
    assert scenarios.curve_node_count == 3
    np.testing.assert_allclose(scenarios.probabilities, [2 / 3, 1 / 3])
    assert source_rates.flags.writeable
    assert source_equity.flags.writeable
    source_rates[0, 0, 0] = 1.0
    source_equity[0, 0] = 0.5
    assert scenarios.rate_shocks[0, 0, 0] == 0
    assert scenarios.equity_returns[0, 0] == pytest.approx(0.1)
    assert scenarios.rate_scenarios(1).shocks.shape == (2, 3)
    with pytest.raises(ValueError, match="greater than -1"):
        qfin.EconomicScenarioSet(np.zeros((1, 1, 3)), equity_returns=-1.0)
    with pytest.raises(ValueError, match="curve node"):
        scenarios.validate_curve(qfin.YieldCurve([0, 1], [0.02, 0.03]))


def test_correlated_economic_scenarios_are_seeded_and_explicit() -> None:
    correlation = np.eye(6)
    standard_deviations = [0.01, 0.002, 0.10, 0.02, 0.05, 0.10]
    first = qfin.EconomicScenarioSet.correlated_gaussian(
        _curve(),
        8,
        4,
        correlation=correlation,
        standard_deviations=standard_deviations,
        seed=19,
        antithetic=True,
    )
    second = qfin.EconomicScenarioSet.correlated_gaussian(
        _curve(),
        8,
        4,
        correlation=correlation,
        standard_deviations=standard_deviations,
        seed=19,
        antithetic=True,
    )
    np.testing.assert_array_equal(first.rate_shocks, second.rate_shocks)
    np.testing.assert_array_equal(first.equity_returns, second.equity_returns)
    assert first.rate_shocks.shape == (8, 4, 3)
    assert np.all(first.mortality_multipliers > 0)
    assert "Gaussian" in first.dependence_assumption


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"probabilities": [0, 0]}, "probabilities"),
        ({"mortality_multipliers": -0.1}, "non-negative"),
        ({"lapse_multipliers": float("nan")}, "finite"),
        ({"labels": ("duplicate", "duplicate")}, "labels"),
    ],
)
def test_economic_scenario_set_rejects_malformed_factors(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        qfin.EconomicScenarioSet(np.zeros((2, 1, 3)), **kwargs)  # type: ignore[arg-type]
