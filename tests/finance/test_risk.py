import numpy as np
import pytest

import qfin


def test_weighted_var_and_expected_shortfall_boundary_mass() -> None:
    distribution = qfin.LossDistribution(
        np.array([-1.0, 0.0, 10.0, 20.0]),
        np.array([0.1, 0.6, 0.2, 0.1]),
    )
    summary = qfin.aggregate_risk(distribution, confidence=0.8, engine="numpy")
    assert summary.mean == pytest.approx(3.9)
    assert summary.var == 10.0
    assert summary.cvar == pytest.approx(15.0)
    assert summary.minimum == -1.0
    assert summary.maximum == 20.0


def test_risk_auto_stays_on_stable_numpy_path() -> None:
    summary = qfin.aggregate_risk(qfin.LossDistribution([0.0, 1.0, 2.0]))
    assert summary.engine == "numpy"


def test_loss_distribution_normalizes_and_maps_to_empirical() -> None:
    distribution = qfin.LossDistribution([1.0, 2.0], [2.0, 2.0])
    assert distribution.probabilities is not None
    np.testing.assert_allclose(distribution.probabilities, [0.5, 0.5])
    assert distribution.as_empirical().mean == pytest.approx(1.5)
    extreme = qfin.LossDistribution([1.0, 2.0], [1.0e308, 1.0e308])
    np.testing.assert_allclose(extreme.probabilities, [0.5, 0.5])


def test_risk_rejects_unrepresentable_moments() -> None:
    distribution = qfin.LossDistribution([-1.0e308, 1.0e308])
    for engine in ("numpy", "native"):
        with pytest.raises(ValueError, match="finite double range"):
            qfin.aggregate_risk(distribution, confidence=0.5, engine=engine)


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
def test_risk_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        qfin.aggregate_risk(qfin.LossDistribution([1.0]), confidence=confidence)
