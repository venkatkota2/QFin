"""Interest-rate quote and compounding conventions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import exp, expm1, isfinite, log, log1p


class Compounding(StrEnum):
    """Supported interest-rate compounding conventions."""

    CONTINUOUS = "continuous"
    ANNUAL = "annual"
    SEMIANNUAL = "semiannual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    SIMPLE = "simple"

    @classmethod
    def parse(cls, value: Compounding | str) -> Compounding:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "continuous": cls.CONTINUOUS,
            "cont": cls.CONTINUOUS,
            "annual": cls.ANNUAL,
            "annually": cls.ANNUAL,
            "semiannual": cls.SEMIANNUAL,
            "semiannually": cls.SEMIANNUAL,
            "quarterly": cls.QUARTERLY,
            "monthly": cls.MONTHLY,
            "simple": cls.SIMPLE,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"compounding must be one of: {choices}") from exc


def compounding_frequency(compounding: Compounding | str) -> int | None:
    selected = Compounding.parse(compounding)
    return {
        Compounding.ANNUAL: 1,
        Compounding.SEMIANNUAL: 2,
        Compounding.QUARTERLY: 4,
        Compounding.MONTHLY: 12,
    }.get(selected)


def discount_factor(
    rate: float,
    time: float,
    compounding: Compounding | str = Compounding.CONTINUOUS,
) -> float:
    """Convert a rate quote into a discount factor at ``time`` years."""

    if not (isfinite(rate) and isfinite(time)) or time < 0:
        raise ValueError("rate and time must be finite, with time non-negative")
    selected = Compounding.parse(compounding)
    if time == 0:
        return 1.0
    if selected is Compounding.CONTINUOUS:
        return exp(-rate * time)
    if selected is Compounding.SIMPLE:
        base = 1.0 + rate * time
        if base <= 0:
            raise ValueError("simple rate requires 1 + rate * time > 0")
        return 1.0 / base
    frequency = compounding_frequency(selected)
    assert frequency is not None
    base = 1.0 + rate / frequency
    if base <= 0:
        raise ValueError("periodic rate must be greater than its negative frequency")
    return float(base ** (-frequency * time))


def rate_from_discount_factor(
    value: float,
    time: float,
    compounding: Compounding | str = Compounding.CONTINUOUS,
) -> float:
    """Return the rate quote implied by a positive discount factor."""

    if not (isfinite(value) and value > 0 and isfinite(time) and time > 0):
        raise ValueError("discount factor and time must be finite and positive")
    selected = Compounding.parse(compounding)
    if selected is Compounding.CONTINUOUS:
        return -log(value) / time
    if selected is Compounding.SIMPLE:
        return (1.0 / value - 1.0) / time
    frequency = compounding_frequency(selected)
    assert frequency is not None
    return frequency * expm1(-log(value) / (frequency * time))


def convert_rate(
    rate: float,
    from_compounding: Compounding | str,
    to_compounding: Compounding | str,
    *,
    time: float = 1.0,
) -> float:
    """Convert a quote while preserving its discount factor over ``time``."""

    source = Compounding.parse(from_compounding)
    target = Compounding.parse(to_compounding)
    if source is target:
        if not (isfinite(rate) and isfinite(time) and time > 0):
            raise ValueError("rate and time must be finite, with time positive")
        return rate
    return rate_from_discount_factor(discount_factor(rate, time, source), time, target)


def continuous_rate(
    rate: float,
    compounding: Compounding | str,
    *,
    time: float = 1.0,
) -> float:
    """Return the continuously compounded equivalent of a quote."""

    selected = Compounding.parse(compounding)
    if not (isfinite(rate) and isfinite(time)) or time < 0:
        raise ValueError("rate and time must be finite, with time non-negative")
    if selected is Compounding.CONTINUOUS:
        return rate
    if selected is Compounding.SIMPLE:
        if time == 0:
            return rate
        base = 1.0 + rate * time
        if base <= 0:
            raise ValueError("simple rate requires 1 + rate * time > 0")
        return log(base) / time
    frequency = compounding_frequency(selected)
    assert frequency is not None
    base = 1.0 + rate / frequency
    if base <= 0:
        raise ValueError("periodic rate must be greater than its negative frequency")
    return frequency * log1p(rate / frequency)


@dataclass(frozen=True, slots=True)
class RateQuote:
    """A rate value whose compounding convention remains attached."""

    rate: float
    compounding: Compounding = Compounding.CONTINUOUS

    def __init__(
        self,
        rate: float,
        compounding: Compounding | str = Compounding.CONTINUOUS,
    ) -> None:
        if not isfinite(rate):
            raise ValueError("rate must be finite")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "compounding", Compounding.parse(compounding))

    def discount(self, time: float) -> float:
        return discount_factor(self.rate, time, self.compounding)

    def equivalent(
        self,
        compounding: Compounding | str,
        *,
        time: float = 1.0,
    ) -> RateQuote:
        target = Compounding.parse(compounding)
        return RateQuote(convert_rate(self.rate, self.compounding, target, time=time), target)


__all__ = [
    "Compounding",
    "RateQuote",
    "compounding_frequency",
    "continuous_rate",
    "convert_rate",
    "discount_factor",
    "rate_from_discount_factor",
]
