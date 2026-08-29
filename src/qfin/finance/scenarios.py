"""Batched zero-curve scenarios with chunked native and NumPy execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index as integer_index
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin import _native
from qfin.finance.curves import YieldCurve
from qfin.finance.fixed_income import Engine

FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RateScenarioSet:
    """Additive zero-rate shocks aligned to a reusable curve's nodes."""

    shocks: FloatArray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shocks = np.ascontiguousarray(self.shocks, dtype=np.float64)
        if shocks.ndim != 2 or shocks.shape[0] == 0 or shocks.shape[1] == 0:
            raise ValueError("shocks must be a non-empty scenario-by-curve-node matrix")
        if not np.all(np.isfinite(shocks)):
            raise ValueError("scenario shocks must be finite")
        labels = self.labels or tuple(f"scenario_{index}" for index in range(shocks.shape[0]))
        if len(labels) != shocks.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("labels must be unique with one label per scenario")
        shocks.setflags(write=False)
        object.__setattr__(self, "shocks", shocks)
        object.__setattr__(self, "labels", tuple(labels))

    @classmethod
    def parallel(cls, curve: YieldCurve, shifts: ArrayLike) -> RateScenarioSet:
        """Construct parallel zero-rate scenarios from decimal shifts."""

        values = np.asarray(shifts, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("parallel shifts must contain finite values")
        matrix = np.repeat(values[:, None], curve.times.size, axis=1)
        labels = tuple(
            f"parallel_{index}_{shift:+.4f}bp"
            for index, shift in enumerate(10_000 * values)
        )
        return cls(matrix, labels)

    @classmethod
    def steepener(
        cls,
        curve: YieldCurve,
        *,
        short_shift: float,
        long_shift: float,
        label: str = "steepener",
    ) -> RateScenarioSet:
        """Construct one linear short-to-long zero-rate twist."""

        if not (isfinite(short_shift) and isfinite(long_shift)):
            raise ValueError("steepener shifts must be finite")
        if curve.times.size == 1:
            shock = np.array([long_shift], dtype=np.float64)
        else:
            scale = (curve.times - curve.times[0]) / (curve.times[-1] - curve.times[0])
            shock = short_shift + scale * (long_shift - short_shift)
        return cls(shock[None, :], (label,))

    @classmethod
    def key_rate(
        cls,
        curve: YieldCurve,
        *,
        key_time: float,
        shift: float,
        width: float,
        label: str | None = None,
    ) -> RateScenarioSet:
        """Construct a triangular key-rate shock at the curve nodes."""

        if not all(isfinite(value) for value in (key_time, shift, width)):
            raise ValueError("key-rate inputs must be finite")
        if key_time < 0 or width <= 0:
            raise ValueError("key_time must be non-negative and width positive")
        weights = np.maximum(1.0 - np.abs(curve.times - key_time) / width, 0.0)
        name = label or f"key_{key_time:g}y_{shift * 10_000:+.1f}bp"
        return cls((shift * weights)[None, :], (name,))


def _validate_flat_portfolio(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    weights: FloatArray,
    curve: YieldCurve,
    scenarios: RateScenarioSet,
) -> None:
    if times.ndim != 1 or amounts.shape != times.shape:
        raise ValueError("cash-flow times and amounts must be one-dimensional and aligned")
    if offsets.ndim != 1 or offsets.size != weights.size + 1:
        raise ValueError("offsets must delimit one cash-flow stream per position weight")
    if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != times.size:
        raise ValueError("invalid cash-flow offsets")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("cash-flow offsets must be non-decreasing")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("cash-flow times must be finite and non-negative")
    if not np.all(np.isfinite(amounts)) or not np.all(np.isfinite(weights)):
        raise ValueError("cash-flow amounts and position weights must be finite")
    if scenarios.shocks.shape[1] != curve.times.size:
        raise ValueError("each scenario must have one shock per curve node")


def _numpy_scenario_values(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    weights: FloatArray,
    curve: YieldCurve,
    shocks: FloatArray,
) -> FloatArray:
    if times.size == 0:
        return np.zeros(shocks.shape[0], dtype=np.float64)
    if curve.times.size == 1:
        interpolated_shocks = np.broadcast_to(shocks[:, :1], (shocks.shape[0], times.size))
    else:
        upper = np.searchsorted(curve.times, times, side="right")
        upper = np.clip(upper, 1, curve.times.size - 1)
        lower = upper - 1
        denominator = curve.times[upper] - curve.times[lower]
        fractions = np.divide(
            times - curve.times[lower],
            denominator,
            out=np.zeros_like(times),
            where=denominator != 0,
        )
        before = times <= curve.times[0]
        after = times >= curve.times[-1]
        lower[before] = 0
        upper[before] = 0
        fractions[before] = 0.0
        lower[after] = curve.times.size - 1
        upper[after] = curve.times.size - 1
        fractions[after] = 0.0
        interpolated_shocks = shocks[:, lower] + fractions * (
            shocks[:, upper] - shocks[:, lower]
        )
    base_rates = np.asarray(curve.zero_rate(times), dtype=np.float64)
    discounted = amounts * np.exp(-(base_rates + interpolated_shocks) * times)
    counts = np.diff(offsets)
    position_index = np.repeat(np.arange(weights.size), counts)
    cashflow_weights = weights[position_index]
    return np.asarray(
        np.sum(discounted * cashflow_weights, axis=1, dtype=np.float64),
        dtype=np.float64,
    )


def scenario_portfolio_values(
    cashflow_times: FloatArray,
    cashflow_amounts: FloatArray,
    offsets: Int64Array,
    position_weights: FloatArray,
    curve: YieldCurve,
    scenarios: RateScenarioSet,
    *,
    engine: Engine = "auto",
    chunk_size: int = 1_024,
) -> tuple[FloatArray, Literal["numpy", "native"]]:
    """Value flattened streams under rate scenarios without materializing a full cube."""

    times = np.ascontiguousarray(cashflow_times, dtype=np.float64).reshape(-1)
    amounts = np.ascontiguousarray(cashflow_amounts, dtype=np.float64).reshape(-1)
    stream_offsets = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
    weights = np.ascontiguousarray(position_weights, dtype=np.float64).reshape(-1)
    _validate_flat_portfolio(times, amounts, stream_offsets, weights, curve, scenarios)
    try:
        normalized_chunk_size = integer_index(chunk_size)
    except TypeError as exc:
        raise ValueError("chunk_size must be a positive integer") from exc
    if isinstance(chunk_size, bool) or normalized_chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    workload = times.size * scenarios.shocks.shape[0]
    selected: Literal["numpy", "native"]
    if engine == "native":
        _native.require()
        selected = "native"
    elif engine == "numpy":
        selected = "numpy"
    else:
        selected = "native" if _native.available() and workload >= 50_000 else "numpy"

    values = np.empty(scenarios.shocks.shape[0], dtype=np.float64)
    for start in range(0, scenarios.shocks.shape[0], normalized_chunk_size):
        stop = min(start + normalized_chunk_size, scenarios.shocks.shape[0])
        shock_chunk = np.ascontiguousarray(scenarios.shocks[start:stop])
        if selected == "native":
            raw = cast(
                object,
                _native.require().scenario_portfolio_present_values(
                    times,
                    amounts,
                    stream_offsets,
                    weights,
                    curve.times,
                    curve.zero_rates,
                    shock_chunk,
                ),
            )
            values[start:stop] = np.asarray(raw, dtype=np.float64)
        else:
            values[start:stop] = _numpy_scenario_values(
                times, amounts, stream_offsets, weights, curve, shock_chunk
            )
    return values, selected


__all__ = ["RateScenarioSet", "scenario_portfolio_values"]
