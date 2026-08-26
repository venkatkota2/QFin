"""Market-model inputs."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BlackScholes:
    """Constant-parameter Black-Scholes market model.

    Parameters are continuously compounded and expressed in annual units.
    """

    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        values = (self.spot, self.rate, self.volatility, self.dividend_yield)
        if not all(isfinite(value) for value in values):
            raise ValueError("Black-Scholes parameters must be finite")
        if self.spot <= 0:
            raise ValueError("spot must be greater than zero")
        if self.volatility <= 0:
            raise ValueError("volatility must be greater than zero")
