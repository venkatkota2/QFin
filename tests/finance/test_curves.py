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
