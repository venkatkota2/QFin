import numpy as np
import pytest

import qfin


def test_gaussian_factor_model_reproduces_dependence_assumption() -> None:
    model = qfin.GaussianFactorModel(
        factor_names=("rates", "equity"),
        correlation=np.array([[1.0, -0.6], [-0.6, 1.0]]),
        means=np.array([0.01, -0.02]),
        standard_deviations=np.array([0.02, 0.15]),
    )
    scenarios = model.simulate(40_000, seed=7)

    np.testing.assert_allclose(np.mean(scenarios.values, axis=0), [0.01, -0.02], atol=0.002)
    np.testing.assert_allclose(np.std(scenarios.values, axis=0), [0.02, 0.15], rtol=0.03)
    assert np.corrcoef(scenarios.values.T)[0, 1] == pytest.approx(-0.6, abs=0.02)
    assert scenarios.dependence_assumption.startswith("Gaussian")


def test_factor_scenarios_map_to_reproducible_linear_losses() -> None:
    model = qfin.GaussianFactorModel(
        ("rates", "equity"),
        np.array([[1.0, 0.25], [0.25, 1.0]]),
    )
    first = model.simulate(10, seed=11, antithetic=True)
    second = model.simulate(10, seed=11, antithetic=True)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_allclose(np.mean(first.values, axis=0), 0.0, atol=1e-15)

    losses = first.linear_loss_distribution([100.0, -50.0], intercept=3.0)
    np.testing.assert_allclose(losses.losses, 3.0 + first.values @ [100.0, -50.0])


@pytest.mark.parametrize(
    "correlation, message",
    [
        (np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        (np.array([[1.0, 1.1], [1.1, 1.0]]), "correlations"),
        (np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]), "semidefinite"),
    ],
)
def test_factor_model_rejects_invalid_correlation(correlation: np.ndarray, message: str) -> None:
    names = tuple(f"factor_{index}" for index in range(correlation.shape[0]))
    with pytest.raises(ValueError, match=message):
        qfin.GaussianFactorModel(names, correlation)
