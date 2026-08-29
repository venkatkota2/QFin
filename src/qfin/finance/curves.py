"""Continuously compounded yield-curve abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class YieldCurve:
    """Zero-rate curve with linear interpolation and flat extrapolation.

    Times are year fractions from the valuation date and rates are continuously
    compounded annual rates. Negative rates are supported.
    """

    times: FloatArray
    zero_rates: FloatArray
    interpolation: Literal["linear"] = "linear"
    extrapolation: Literal["flat"] = "flat"

    def __post_init__(self) -> None:
        times = np.array(self.times, dtype=np.float64, order="C", copy=True).reshape(-1)
        rates = np.array(self.zero_rates, dtype=np.float64, order="C", copy=True).reshape(-1)
        if times.size == 0 or times.shape != rates.shape:
            raise ValueError("times and zero_rates must have equal non-zero length")
        if not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("curve times must be finite and non-negative")
        if not np.all(np.isfinite(rates)):
            raise ValueError("zero rates must be finite")
        if np.any(np.diff(times) <= 0):
            raise ValueError("curve times must be strictly increasing")
        if self.interpolation != "linear":
            raise ValueError("only linear interpolation is currently implemented")
        if self.extrapolation != "flat":
            raise ValueError("only flat extrapolation is currently implemented")
        times.setflags(write=False)
        rates.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "zero_rates", rates)

    @overload
    def zero_rate(self, time: float) -> float: ...

    @overload
    def zero_rate(self, time: ArrayLike) -> FloatArray: ...

    def zero_rate(self, time: float | ArrayLike) -> float | FloatArray:
        """Interpolate zero rates at one or many non-negative times."""

        query = np.asarray(time, dtype=np.float64)
        if not np.all(np.isfinite(query)) or np.any(query < 0):
            raise ValueError("query times must be finite and non-negative")
        values = np.interp(
            query,
            self.times,
            self.zero_rates,
            left=self.zero_rates[0],
            right=self.zero_rates[-1],
        )
        if query.ndim == 0:
            return float(values)
        return np.asarray(values, dtype=np.float64)

    @overload
    def discount(self, time: float) -> float: ...

    @overload
    def discount(self, time: ArrayLike) -> FloatArray: ...

    def discount(self, time: float | ArrayLike) -> float | FloatArray:
        """Return ``exp(-r(t) t)`` for one or many times."""

        query = np.asarray(time, dtype=np.float64)
        rates = np.asarray(self.zero_rate(query), dtype=np.float64)
        discounts = np.exp(-rates * query)
        if query.ndim == 0:
            return float(discounts)
        return np.asarray(discounts, dtype=np.float64)

    def forward_rate(self, start: float, end: float) -> float:
        """Return the continuously compounded forward rate on ``[start, end]``."""

        if not (isfinite(start) and isfinite(end)) or start < 0 or end <= start:
            raise ValueError("require finite times with 0 <= start < end")
        start_rate = self.zero_rate(start)
        end_rate = self.zero_rate(end)
        return (end * end_rate - start * start_rate) / (end - start)

    def shifted(self, shift: float | ArrayLike) -> YieldCurve:
        """Return a curve with an additive scalar or node-by-node rate shock."""

        shifts = np.asarray(shift, dtype=np.float64)
        if not np.all(np.isfinite(shifts)):
            raise ValueError("curve shifts must be finite")
        if shifts.ndim > 1 or (shifts.ndim == 1 and shifts.shape != self.zero_rates.shape):
            raise ValueError("shift must be scalar or have one value per curve node")
        return YieldCurve(
            times=self.times,
            zero_rates=np.asarray(self.zero_rates + shifts, dtype=np.float64),
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
        )


__all__ = ["YieldCurve"]
