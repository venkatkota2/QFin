"""Classical reference calculations for quantum-result validation."""

from math import exp, log, sqrt

from scipy.special import ndtr

from qfin.finance import BlackScholes, EuropeanOption


def black_scholes_price(option: EuropeanOption, market: BlackScholes) -> float:
    """Return the analytical Black-Scholes value of a European option."""

    time = option.maturity
    volatility_time = market.volatility * sqrt(time)
    d1 = (
        log(market.spot / option.strike)
        + (
            market.rate
            - market.dividend_yield
            + 0.5 * market.volatility**2
        )
        * time
    ) / volatility_time
    d2 = d1 - volatility_time
    discounted_spot = market.spot * exp(-market.dividend_yield * time)
    discounted_strike = option.strike * exp(-market.rate * time)
    if option.kind == "call":
        return float(discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2))
    return float(discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1))


def put_call_parity_residual(
    call_value: float,
    put_value: float,
    strike: float,
    maturity: float,
    market: BlackScholes,
) -> float:
    """Return zero when European call and put values satisfy parity."""

    right_hand_side = market.spot * exp(-market.dividend_yield * maturity)
    right_hand_side -= strike * exp(-market.rate * maturity)
    return call_value - put_value - right_hand_side
