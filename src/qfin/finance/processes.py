"""Stochastic-process mappings used by the compiler."""

from dataclasses import dataclass
from math import log, sqrt

from qfin.finance.distributions import LogNormal
from qfin.finance.models import BlackScholes


@dataclass(frozen=True, slots=True)
class GeometricBrownianMotion:
    """Risk-neutral geometric Brownian motion for a Black-Scholes market."""

    market: BlackScholes

    def terminal_distribution(self, horizon: float) -> LogNormal:
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero")
        variance_drift = 0.5 * self.market.volatility**2
        mu = log(self.market.spot) + (
            self.market.rate - self.market.dividend_yield - variance_drift
        ) * horizon
        sigma = self.market.volatility * sqrt(horizon)
        return LogNormal(mu=mu, sigma=sigma)
