"""Vectorized fixed-income cash-flow and valuation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _readonly_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least one finite value")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class DiscountCurve:
    """Continuously compounded zero curve with linear rate interpolation."""

    times: FloatArray
    zero_rates: FloatArray

    def __post_init__(self) -> None:
        times = _readonly_vector(self.times, name="times")
        rates = _readonly_vector(self.zero_rates, name="zero_rates")
        if times.shape != rates.shape:
            raise ValueError("times and zero_rates must have the same shape")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("curve times must be non-negative and strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "zero_rates", rates)

    @classmethod
    def flat(cls, rate: float, *, horizon: float = 100.0) -> DiscountCurve:
        """Construct a flat continuously compounded curve."""
        if not isfinite(rate):
            raise ValueError("rate must be finite")
        if not isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be finite and positive")
        return cls(
            times=np.array([0.0, horizon], dtype=np.float64),
            zero_rates=np.array([rate, rate], dtype=np.float64),
        )

    def zero_rate(self, times: ArrayLike) -> FloatArray:
        query = np.asarray(times, dtype=np.float64)
        if np.any(~np.isfinite(query)) or np.any(query < 0):
            raise ValueError("discount times must be finite and non-negative")
        return np.asarray(
            np.interp(query, self.times, self.zero_rates), dtype=np.float64
        )

    def discount(self, times: ArrayLike) -> FloatArray:
        query = np.asarray(times, dtype=np.float64)
        return np.asarray(np.exp(-self.zero_rate(query) * query), dtype=np.float64)

    def discount_factor(self, time: float) -> float:
        return float(self.discount(np.array([time], dtype=np.float64))[0])

    def shifted(self, parallel_shift: float) -> DiscountCurve:
        """Return a curve shifted by a decimal rate amount (0.01 = 100 bp)."""
        if not isfinite(parallel_shift):
            raise ValueError("parallel_shift must be finite")
        return DiscountCurve(self.times, self.zero_rates + parallel_shift)


@dataclass(frozen=True, slots=True)
class CashFlowSchedule:
    """Sorted, date-aggregated deterministic cash flows."""

    times: FloatArray
    amounts: FloatArray

    def __post_init__(self) -> None:
        times = _readonly_vector(self.times, name="cash-flow times")
        amounts = _readonly_vector(self.amounts, name="cash-flow amounts")
        if times.shape != amounts.shape:
            raise ValueError("cash-flow times and amounts must have the same shape")
        if np.any(times < 0):
            raise ValueError("cash-flow times must be non-negative")
        order = np.argsort(times, kind="stable")
        sorted_times = times[order]
        sorted_amounts = amounts[order]
        unique_times, starts = np.unique(sorted_times, return_index=True)
        aggregated = np.add.reduceat(sorted_amounts, starts)
        nonzero = aggregated != 0.0
        if np.any(nonzero):
            unique_times = unique_times[nonzero]
            aggregated = aggregated[nonzero]
        else:
            unique_times = unique_times[:1]
            aggregated = aggregated[:1]
        unique_times.setflags(write=False)
        aggregated.setflags(write=False)
        object.__setattr__(self, "times", unique_times)
        object.__setattr__(self, "amounts", aggregated)

    @classmethod
    def combine(cls, *schedules: CashFlowSchedule) -> CashFlowSchedule:
        if not schedules:
            raise ValueError("at least one schedule is required")
        return cls(
            times=np.concatenate([schedule.times for schedule in schedules]),
            amounts=np.concatenate([schedule.amounts for schedule in schedules]),
        )

    def scaled(self, multiplier: float) -> CashFlowSchedule:
        if not isfinite(multiplier):
            raise ValueError("multiplier must be finite")
        return CashFlowSchedule(self.times, self.amounts * multiplier)

    def present_value(self, curve: DiscountCurve) -> float:
        return float(np.dot(self.amounts, curve.discount(self.times)))

    def parallel_duration(self, curve: DiscountCurve) -> float:
        discounted = self.amounts * curve.discount(self.times)
        present_value = float(np.sum(discounted))
        if abs(present_value) <= 1e-15:
            raise ValueError("duration is undefined for a zero present value")
        return float(np.dot(self.times, discounted) / present_value)

    def parallel_convexity(self, curve: DiscountCurve) -> float:
        discounted = self.amounts * curve.discount(self.times)
        present_value = float(np.sum(discounted))
        if abs(present_value) <= 1e-15:
            raise ValueError("convexity is undefined for a zero present value")
        return float(np.dot(self.times**2, discounted) / present_value)


@dataclass(frozen=True, slots=True)
class FixedRateBond:
    """Bullet fixed-rate bond valued from deterministic contractual cash flows."""

    face_value: float
    coupon_rate: float
    maturity: float
    coupon_frequency: int = 2

    def __post_init__(self) -> None:
        values = (self.face_value, self.coupon_rate, self.maturity)
        if not all(isfinite(value) for value in values):
            raise ValueError("bond inputs must be finite")
        if self.face_value <= 0 or self.maturity <= 0:
            raise ValueError("face_value and maturity must be positive")
        if self.coupon_rate < 0:
            raise ValueError("coupon_rate must be non-negative")
        if isinstance(self.coupon_frequency, bool) or self.coupon_frequency < 1:
            raise ValueError("coupon_frequency must be a positive integer")
        periods = self.maturity * self.coupon_frequency
        if not np.isclose(periods, round(periods), atol=1e-10):
            raise ValueError("maturity must contain a whole number of coupon periods")

    def cashflows(self) -> CashFlowSchedule:
        periods = round(self.maturity * self.coupon_frequency)
        times = np.arange(1, periods + 1, dtype=np.float64) / self.coupon_frequency
        amounts = np.full(
            periods,
            self.face_value * self.coupon_rate / self.coupon_frequency,
            dtype=np.float64,
        )
        amounts[-1] += self.face_value
        return CashFlowSchedule(times, amounts)

    def price(self, curve: DiscountCurve) -> float:
        return self.cashflows().present_value(curve)

    def duration(self, curve: DiscountCurve) -> float:
        return self.cashflows().parallel_duration(curve)

    def convexity(self, curve: DiscountCurve) -> float:
        return self.cashflows().parallel_convexity(curve)


@dataclass(frozen=True, slots=True)
class BondPosition:
    instrument: FixedRateBond
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.quantity) or self.quantity < 0:
            raise ValueError("bond quantity must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FixedIncomePortfolio:
    """Fixed-income positions with a pre-aggregated cash-flow vector."""

    positions: tuple[BondPosition, ...]
    _cashflows: CashFlowSchedule = field(init=False, repr=False)

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        if not positions:
            raise ValueError("fixed-income portfolio requires at least one position")
        schedules = tuple(
            position.instrument.cashflows().scaled(position.quantity)
            for position in positions
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "_cashflows", CashFlowSchedule.combine(*schedules))

    @property
    def cashflows(self) -> CashFlowSchedule:
        return self._cashflows

    def present_value(self, curve: DiscountCurve) -> float:
        return self._cashflows.present_value(curve)

    def duration(self, curve: DiscountCurve) -> float:
        return self._cashflows.parallel_duration(curve)

    def convexity(self, curve: DiscountCurve) -> float:
        return self._cashflows.parallel_convexity(curve)
