"""Interest-rate quote and compounding conventions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import exp, expm1, isfinite, log, log1p

from qfin.exceptions import QFinValidationError


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
            raise QFinValidationError(f"compounding must be one of: {choices}") from exc


def compounding_frequency(compounding: Compounding | str) -> int | None:
    selected = Compounding.parse(compounding)
    return {
        Compounding.ANNUAL: 1,
        Compounding.SEMIANNUAL: 2,
        Compounding.QUARTERLY: 4,
        Compounding.MONTHLY: 12,
    }.get(selected)


def _finite_result(value: float, *, positive: bool = False) -> float:
    if not isfinite(value) or (positive and value <= 0):
        raise QFinValidationError("rate conversion exceeds the finite double range")
    return value


def _exp_result(value: float, *, subtract_one: bool = False) -> float:
    try:
        return _finite_result(
            expm1(value) if subtract_one else exp(value), positive=not subtract_one
        )
    except OverflowError as exc:
        raise QFinValidationError("rate conversion exceeds the finite double range") from exc


def discount_factor(
    rate: float,
    time: float,
    compounding: Compounding | str = Compounding.CONTINUOUS,
) -> float:
    """Convert a rate quote into a discount factor at ``time`` years."""

    if not (isfinite(rate) and isfinite(time)) or time < 0:
        raise QFinValidationError("rate and time must be finite, with time non-negative")
    selected = Compounding.parse(compounding)
    if time == 0:
        return 1.0
    if selected is Compounding.CONTINUOUS:
        return _exp_result(-rate * time)
    if selected is Compounding.SIMPLE:
        base = 1.0 + rate * time
        if base <= 0:
            raise QFinValidationError("simple rate requires 1 + rate * time > 0")
        return _finite_result(1.0 / base, positive=True)
    frequency = compounding_frequency(selected)
    assert frequency is not None
    base = 1.0 + rate / frequency
    if base <= 0:
        raise QFinValidationError("periodic rate must be greater than its negative frequency")
    return _exp_result(-time * (frequency * log1p(rate / frequency)))


def rate_from_discount_factor(
    value: float,
    time: float,
    compounding: Compounding | str = Compounding.CONTINUOUS,
) -> float:
    """Return the rate quote implied by a positive discount factor."""

    if not (isfinite(value) and value > 0 and isfinite(time) and time > 0):
        raise QFinValidationError("discount factor and time must be finite and positive")
    selected = Compounding.parse(compounding)
    if selected is Compounding.CONTINUOUS:
        return _finite_result(-log(value) / time)
    if selected is Compounding.SIMPLE:
        return _finite_result(_exp_result(-log(value), subtract_one=True) / time)
    frequency = compounding_frequency(selected)
    assert frequency is not None
    return _finite_result(
        frequency * _exp_result((-log(value) / time) / frequency, subtract_one=True)
    )


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
    if not (isfinite(rate) and isfinite(time) and time > 0):
        raise QFinValidationError("rate and time must be finite, with time positive")
    continuous = continuous_rate(rate, source, time=time)
    if source is target:
        return rate
    if target is Compounding.CONTINUOUS:
        return continuous
    if target is Compounding.SIMPLE:
        return _finite_result(_exp_result(continuous * time, subtract_one=True) / time)
    frequency = compounding_frequency(target)
    assert frequency is not None
    return _finite_result(frequency * _exp_result(continuous / frequency, subtract_one=True))


def continuous_rate(
    rate: float,
    compounding: Compounding | str,
    *,
    time: float = 1.0,
) -> float:
    """Return the continuously compounded equivalent of a quote."""

    selected = Compounding.parse(compounding)
    if not (isfinite(rate) and isfinite(time)) or time < 0:
        raise QFinValidationError("rate and time must be finite, with time non-negative")
    if selected is Compounding.CONTINUOUS:
        return rate
    if selected is Compounding.SIMPLE:
        if time == 0:
            return rate
        base = 1.0 + rate * time
        if base <= 0:
            raise QFinValidationError("simple rate requires 1 + rate * time > 0")
        return _finite_result(log1p(rate * time) / time)
    frequency = compounding_frequency(selected)
    assert frequency is not None
    base = 1.0 + rate / frequency
    if base <= 0:
        raise QFinValidationError("periodic rate must be greater than its negative frequency")
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
            raise QFinValidationError("rate must be finite")
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
