import numpy as np
import pytest

import qfin


@pytest.fixture
def alm_model() -> qfin.AssetLiabilityModel:
    mortality = qfin.MortalityTable(
        ages=np.arange(40, 101),
        qx=np.concatenate([np.full(60, 0.02), np.array([1.0])]),
    )
    policies = qfin.LifePolicyPortfolio(
        (
            qfin.PolicyPosition(
                qfin.TermLifePolicy(
                    issue_age=40,
                    term=30,
                    face_amount=100_000,
                    annual_premium=500,
                ),
                count=100,
            ),
        )
    )
    assets = qfin.FixedIncomePortfolio(
        (
            qfin.BondPosition(
                qfin.FixedRateBond(1_000, 0.04, 10.0, coupon_frequency=2),
                quantity=2_000,
            ),
        )
    )
    return qfin.AssetLiabilityModel(
        assets=assets,
        liabilities=policies,
        discount_curve=qfin.DiscountCurve.flat(0.04),
        mortality=mortality,
    )


def test_alm_base_valuation_contains_matching_metrics(
    alm_model: qfin.AssetLiabilityModel,
) -> None:
    result = alm_model.evaluate()
    assert result.asset_value > 0
    assert result.liability_value > 0
    assert result.surplus == pytest.approx(result.asset_value - result.liability_value)
    assert result.funding_ratio == pytest.approx(
        result.asset_value / result.liability_value
    )
    assert result.duration_gap == pytest.approx(
        result.asset_duration - result.liability_duration
    )


def test_vectorized_scenarios_match_individual_curve_valuations(
    alm_model: qfin.AssetLiabilityModel,
) -> None:
    shocks = np.array([-0.03, -0.01, 0.0, 0.02, 0.04])
    scenarios = alm_model.run_parallel_shocks(
        shocks,
        probabilities=np.array([1, 2, 3, 2, 2]),
        max_working_bytes=64,
    )
    for index, shock in enumerate(shocks):
        individual = alm_model.evaluate(alm_model.discount_curve.shifted(float(shock)))
        assert scenarios.asset_values[index] == pytest.approx(individual.asset_value)
        assert scenarios.liability_values[index] == pytest.approx(
            individual.liability_value
        )
    assert np.sum(scenarios.probabilities) == pytest.approx(1.0)
    assert scenarios.expected_shortfall == pytest.approx(
        np.dot(scenarios.probabilities, np.maximum(-scenarios.surplus, 0.0))
    )
    assert 0 <= scenarios.shortfall_probability <= 1


def test_scenario_chunk_size_does_not_change_results(
    alm_model: qfin.AssetLiabilityModel,
) -> None:
    shocks = np.linspace(-0.05, 0.05, 257)
    small = alm_model.run_parallel_shocks(shocks, max_working_bytes=128)
    large = alm_model.run_parallel_shocks(shocks, max_working_bytes=64 * 1024 * 1024)
    assert small.asset_values == pytest.approx(large.asset_values)
    assert small.liability_values == pytest.approx(large.liability_values)
    assert not small.surplus.flags.writeable


def test_scenario_probabilities_are_validated(
    alm_model: qfin.AssetLiabilityModel,
) -> None:
    with pytest.raises(ValueError, match="match"):
        alm_model.run_parallel_shocks([0.0, 0.01], probabilities=[1.0])
