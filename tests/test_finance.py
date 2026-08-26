from math import exp

import numpy as np
import pytest

import qfin


def test_black_scholes_put_call_parity() -> None:
    market = qfin.BlackScholes(
        spot=100.0,
        rate=0.04,
        volatility=0.20,
        dividend_yield=0.01,
    )
    call = qfin.EuropeanCall(strike=105.0, maturity=1.25)
    put = qfin.EuropeanPut(strike=105.0, maturity=1.25)
    call_value = qfin.black_scholes_price(call, market)
    put_value = qfin.black_scholes_price(put, market)
    expected = market.spot * exp(-market.dividend_yield * call.maturity)
    expected -= call.strike * exp(-market.rate * call.maturity)
    assert call_value - put_value == pytest.approx(expected, abs=1e-12)


def test_payoffs_are_vectorized() -> None:
    terminal = np.array([80.0, 100.0, 120.0])
    call = qfin.EuropeanCall(strike=100.0, maturity=1.0)
    put = qfin.EuropeanPut(strike=100.0, maturity=1.0)
    np.testing.assert_array_equal(call.payoff(terminal), [0.0, 0.0, 20.0])
    np.testing.assert_array_equal(put.payoff(terminal), [20.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: qfin.BlackScholes(0, 0.04, 0.2), "spot"),
        (lambda: qfin.BlackScholes(100, 0.04, 0), "volatility"),
        (lambda: qfin.EuropeanCall(0, 1), "strike"),
        (lambda: qfin.EuropeanPut(100, 0), "maturity"),
    ],
)
def test_invalid_financial_inputs_fail_early(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_empirical_distribution_normalizes_and_combines_duplicates() -> None:
    distribution = qfin.EmpiricalDistribution(
        values=np.array([2.0, 1.0, 2.0]),
        probabilities=np.array([0.2, 0.5, 0.3]),
    )
    np.testing.assert_array_equal(distribution.values, [1.0, 2.0])
    assert distribution.probabilities is not None
    np.testing.assert_allclose(distribution.probabilities, [0.5, 0.5])
    assert distribution.mean == pytest.approx(1.5)

