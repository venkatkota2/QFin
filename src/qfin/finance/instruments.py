"""Financial instruments supported by the QFin MVP."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_option(strike: float, maturity: float) -> None:
    if not isfinite(strike) or strike <= 0:
        raise ValueError("strike must be finite and greater than zero")
    if not isfinite(maturity) or maturity <= 0:
        raise ValueError("maturity must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class EuropeanCall:
    """European call option with payoff ``max(S_T - K, 0)``."""

    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _validate_option(self.strike, self.maturity)

    @property
    def kind(self) -> Literal["call"]:
        return "call"

    def payoff(self, terminal_value: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(terminal_value, dtype=np.float64)
        return np.maximum(values - self.strike, 0.0)


@dataclass(frozen=True, slots=True)
class EuropeanPut:
    """European put option with payoff ``max(K - S_T, 0)``."""

    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _validate_option(self.strike, self.maturity)

    @property
    def kind(self) -> Literal["put"]:
        return "put"

    def payoff(self, terminal_value: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(terminal_value, dtype=np.float64)
        return np.maximum(self.strike - values, 0.0)


EuropeanOption: TypeAlias = EuropeanCall | EuropeanPut
