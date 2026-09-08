import numpy as np
import pytest

import qfin


@pytest.mark.parametrize("scale", [1.0, 1e-308, 1e308])
def test_probability_normalization_is_scale_invariant_across_modules(scale: float) -> None:
    weights = np.array([1.0, 0.5, 0.5]) * scale
    original = weights.copy()
    empirical = qfin.EmpiricalDistribution(np.array([0.0, 1.0, 2.0]), weights)
    risk = qfin.LossDistribution([0.0, 1.0, 2.0], weights)
    scenarios = qfin.EconomicScenarioSet(np.zeros((3, 1)), probabilities=weights)
    for probabilities in (empirical.probabilities, risk.probabilities, scenarios.probabilities):
        np.testing.assert_allclose(probabilities, [0.5, 0.25, 0.25], rtol=1e-15)
        assert np.sum(probabilities) == pytest.approx(1.0, abs=2e-16)
    assert empirical.mean == pytest.approx(0.75)
    assert empirical.cdf(2.0) == pytest.approx(1.0)
    np.testing.assert_array_equal(weights, original)
    assert weights.flags.writeable


def test_empirical_duplicate_and_zero_weight_quantiles() -> None:
    distribution = qfin.EmpiricalDistribution([2.0, 0.0, 1.0, 1.0], [0, 1, 1, 2])
    np.testing.assert_allclose(distribution.cdf([-1, 0, 0.5, 1, 5]), [0, .25, .25, 1, 1])
    np.testing.assert_array_equal(distribution.ppf([0, .25, .26, 1]), [0, 0, 1, 1])
    for invalid in ([-.1], [1.1], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="quantiles"):
            distribution.ppf(invalid)


@pytest.mark.parametrize("values,weights", [
    ([], None), ([np.inf], None), ([1, 2], [1]), ([1, 2], [-1, 1]),
    ([1, 2], [np.nan, 1]), ([1, 2], [0, 0]),
])
def test_empirical_factory_rejects_invalid_distributions(
    values: object, weights: object,
) -> None:
    with pytest.raises(ValueError):
        qfin.EmpiricalDistribution(values, weights)


@pytest.mark.parametrize("distribution", [qfin.Normal(3.0, 2.0), qfin.LogNormal(1.0, 0.3)])
def test_continuous_cdf_quantile_round_trip(distribution: object) -> None:
    quantiles = np.array([.0001, .01, .5, .99, .9999])
    np.testing.assert_allclose(distribution.cdf(distribution.ppf(quantiles)), quantiles,
                               rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("factory,args", [
    (qfin.Normal, (np.nan, 1)), (qfin.Normal, (0, 0)),
    (qfin.LogNormal, (np.inf, 1)), (qfin.LogNormal, (0, -1)),
])
def test_continuous_distribution_parameters_are_finite(factory: object, args: tuple) -> None:
    with pytest.raises(ValueError):
        factory(*args)


@pytest.mark.parametrize("compounding,rate,time", [
    ("annual", -1.0, 1.0), ("semiannual", -2.0, 1.0),
    ("simple", -.1, 10.0),
])
def test_rate_identity_conversion_does_not_bypass_domain_validation(
    compounding: str, rate: float, time: float,
) -> None:
    with pytest.raises(ValueError):
        qfin.convert_rate(rate, compounding, compounding, time=time)
