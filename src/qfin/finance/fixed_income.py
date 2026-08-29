"""Fixed-rate cash flows, batch valuation, yield solving, and rate risk."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from operator import index as integer_index
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin import _native
from qfin.finance.curves import YieldCurve

FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]
Int32Array = NDArray[np.int32]
Engine = Literal["auto", "numpy", "native"]

# Crossing point is benchmarked by examples/native_benchmark.py. This conservative
# default avoids paying extension-boundary overhead for tiny portfolios.
_AUTO_NATIVE_CASHFLOW_THRESHOLD = 4_096


@dataclass(frozen=True, slots=True)
class CashFlow:
    """One deterministic cash flow at a non-negative year fraction."""

    time: float
    amount: float

    def __post_init__(self) -> None:
        if not isfinite(self.time) or self.time < 0:
            raise ValueError("cash-flow time must be finite and non-negative")
        if not isfinite(self.amount):
            raise ValueError("cash-flow amount must be finite")


@dataclass(frozen=True, slots=True)
class FixedRateBond:
    """Fixed-rate bond with regular coupons and an optional final stub period."""

    maturity: float
    coupon_rate: float
    face_value: float = 100.0
    frequency: int = 2

    def __post_init__(self) -> None:
        values = (self.maturity, self.coupon_rate, self.face_value)
        if not all(isfinite(value) for value in values):
            raise ValueError("bond inputs must be finite")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.face_value <= 0:
            raise ValueError("face_value must be positive")
        try:
            frequency = integer_index(self.frequency)
        except TypeError as exc:
            raise ValueError("frequency must be an integer in [1, 365]") from exc
        if isinstance(self.frequency, bool) or frequency <= 0 or frequency > 365:
            raise ValueError("frequency must be an integer in [1, 365]")
        object.__setattr__(self, "frequency", frequency)

    def payment_schedule(self) -> FloatArray:
        """Return coupon/principal payment times measured from issue."""

        periods = floor(self.maturity * self.frequency + 1.0e-12)
        times = np.arange(1, periods + 1, dtype=np.float64) / self.frequency
        if times.size == 0 or times[-1] < self.maturity - 1.0e-12:
            times = np.append(times, self.maturity)
        elif times[-1] > self.maturity + 1.0e-12:
            times[-1] = self.maturity
        return np.asarray(times, dtype=np.float64)

    def cashflows(self, *, settlement: float = 0.0) -> tuple[FloatArray, FloatArray]:
        """Return remaining times and amounts relative to ``settlement``."""

        if not isfinite(settlement) or settlement < 0:
            raise ValueError("settlement must be finite and non-negative")
        payment_times = self.payment_schedule()
        previous_times = np.concatenate((np.array([0.0]), payment_times[:-1]))
        accrual_fractions = payment_times - previous_times
        amounts = self.face_value * self.coupon_rate * accrual_fractions
        amounts[-1] += self.face_value
        remaining = payment_times > settlement + 1.0e-12
        relative_times = np.ascontiguousarray(payment_times[remaining] - settlement)
        remaining_amounts = np.ascontiguousarray(amounts[remaining])
        return relative_times, remaining_amounts

    def accrued_interest(self, settlement: float = 0.0) -> float:
        """Return straight-line coupon accrual from the preceding payment date."""

        if not isfinite(settlement) or settlement < 0:
            raise ValueError("settlement must be finite and non-negative")
        if settlement <= 0 or settlement >= self.maturity:
            return 0.0
        schedule = self.payment_schedule()
        completed = schedule[schedule <= settlement + 1.0e-12]
        previous = 0.0 if completed.size == 0 else float(completed[-1])
        if abs(previous - settlement) <= 1.0e-12:
            return 0.0
        return self.face_value * self.coupon_rate * (settlement - previous)


@dataclass(frozen=True, slots=True)
class BondBatchAnalytics:
    """Vectorized bond valuation and rate-risk outputs."""

    dirty_prices: FloatArray
    clean_prices: FloatArray
    accrued_interest: FloatArray
    macaulay_duration: FloatArray
    modified_duration: FloatArray
    convexity: FloatArray
    dv01: FloatArray
    engine: Literal["numpy", "native"]


@dataclass(frozen=True, slots=True)
class YieldSolveResult:
    """Batch yield-to-maturity solution with convergence diagnostics."""

    yields: FloatArray
    converged: NDArray[np.bool_]
    iterations: Int32Array
    engine: Literal["numpy", "native"]


def _as_bond_tuple(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
) -> tuple[FixedRateBond, ...]:
    items = (bonds,) if isinstance(bonds, FixedRateBond) else tuple(bonds)
    if any(not isinstance(item, FixedRateBond) for item in items):
        raise TypeError("bonds must contain FixedRateBond objects")
    return items


def flatten_bond_cashflows(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    *,
    settlement: float = 0.0,
) -> tuple[FloatArray, FloatArray, Int64Array]:
    """Flatten a bond batch into contiguous buffers and prefix offsets."""

    items = _as_bond_tuple(bonds)
    time_parts: list[FloatArray] = []
    amount_parts: list[FloatArray] = []
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    for index, bond in enumerate(items):
        times, amounts = bond.cashflows(settlement=settlement)
        time_parts.append(times)
        amount_parts.append(amounts)
        offsets[index + 1] = offsets[index] + times.size
    times = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=np.float64)
    amounts = np.concatenate(amount_parts) if amount_parts else np.empty(0, dtype=np.float64)
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(amounts, dtype=np.float64),
        np.ascontiguousarray(offsets, dtype=np.int64),
    )


def _resolve_engine(engine: Engine, workload: int) -> Literal["numpy", "native"]:
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    if engine == "native":
        _native.require()
        return "native"
    if engine == "numpy":
        return "numpy"
    if _native.available() and workload >= _AUTO_NATIVE_CASHFLOW_THRESHOLD:
        return "native"
    return "numpy"


def _segment_sum(values: FloatArray, offsets: Int64Array) -> FloatArray:
    counts = np.diff(offsets)
    indices = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    return np.bincount(indices, weights=values, minlength=counts.size).astype(np.float64)


def _numpy_curve_metrics(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    curve: YieldCurve,
    shift: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rates = np.asarray(curve.zero_rate(times), dtype=np.float64) + shift
    present_values = amounts * np.exp(-rates * times)
    prices = _segment_sum(present_values, offsets)
    first = _segment_sum(times * present_values, offsets)
    second = _segment_sum(times * times * present_values, offsets)
    durations = np.divide(first, prices, out=np.zeros_like(prices), where=prices != 0)
    convexities = np.divide(second, prices, out=np.zeros_like(prices), where=prices != 0)
    down = _segment_sum(amounts * np.exp(-(rates - 1.0e-4) * times), offsets)
    up = _segment_sum(amounts * np.exp(-(rates + 1.0e-4) * times), offsets)
    return prices, durations, convexities, 0.5 * (down - up)


def price_bonds(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    curve: YieldCurve,
    *,
    settlement: float = 0.0,
    parallel_shift: float = 0.0,
    engine: Engine = "auto",
) -> BondBatchAnalytics:
    """Price a batch against a zero curve with one native boundary crossing."""

    if not isfinite(parallel_shift):
        raise ValueError("parallel_shift must be finite")
    items = _as_bond_tuple(bonds)
    times, amounts, offsets = flatten_bond_cashflows(items, settlement=settlement)
    selected = _resolve_engine(engine, times.size)
    if selected == "native":
        raw = cast(
            dict[str, object],
            _native.require().price_cashflow_batches(
                times,
                amounts,
                offsets,
                curve.times,
                curve.zero_rates,
                parallel_shift,
            ),
        )
        prices = np.asarray(raw["prices"], dtype=np.float64)
        durations = np.asarray(raw["macaulay_durations"], dtype=np.float64)
        convexities = np.asarray(raw["convexities"], dtype=np.float64)
        dv01 = np.asarray(raw["dv01"], dtype=np.float64)
    else:
        prices, durations, convexities, dv01 = _numpy_curve_metrics(
            times, amounts, offsets, curve, parallel_shift
        )
    accrued = np.asarray(
        [bond.accrued_interest(settlement) for bond in items], dtype=np.float64
    )
    return BondBatchAnalytics(
        dirty_prices=prices,
        clean_prices=prices - accrued,
        accrued_interest=accrued,
        macaulay_duration=durations,
        modified_duration=durations.copy(),
        convexity=convexities,
        dv01=dv01,
        engine=selected,
    )


def _broadcast_values(values: ArrayLike, count: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(count, float(array), dtype=np.float64)
    else:
        array = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or one value per bond")
    return array


def _numpy_yield_metrics(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    yields: FloatArray,
    frequencies: Int32Array,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    counts = np.diff(offsets)
    repeated_yields = np.repeat(yields, counts)
    repeated_frequencies = np.repeat(frequencies.astype(np.float64), counts)
    base = 1.0 + repeated_yields / repeated_frequencies
    if np.any(base <= 0):
        raise ValueError("yield must be greater than -frequency")
    present_values = amounts * np.power(base, -repeated_frequencies * times)
    prices = _segment_sum(present_values, offsets)
    first = _segment_sum(times * present_values, offsets)
    macaulay = np.divide(first, prices, out=np.zeros_like(prices), where=prices != 0)
    periods = repeated_frequencies * times
    numerator = _segment_sum(periods * (periods + 1.0) * present_values, offsets)
    denominator = prices * frequencies.astype(np.float64) ** 2 * (
        1.0 + yields / frequencies
    ) ** 2
    convexity = np.divide(
        numerator, denominator, out=np.zeros_like(prices), where=denominator != 0
    )

    def prices_at(candidate_yields: FloatArray) -> FloatArray:
        repeated = np.repeat(candidate_yields, counts)
        discount_base = 1.0 + repeated / repeated_frequencies
        values = amounts * np.power(discount_base, -repeated_frequencies * times)
        return _segment_sum(values, offsets)

    up = prices_at(yields + 1.0e-4)
    central = yields - 1.0e-4 > -frequencies
    down_candidates = np.where(central, yields - 1.0e-4, yields)
    down = prices_at(down_candidates)
    dv01 = np.where(central, 0.5 * (down - up), prices - up)
    return prices, macaulay, convexity, dv01


def price_bonds_from_yield(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    yields: ArrayLike,
    *,
    settlement: float = 0.0,
    engine: Engine = "auto",
) -> BondBatchAnalytics:
    """Return price, duration, convexity, and DV01 from nominal annual yields."""

    items = _as_bond_tuple(bonds)
    yield_array = _broadcast_values(yields, len(items), "yields")
    frequencies = np.asarray([bond.frequency for bond in items], dtype=np.int32)
    if np.any(1.0 + yield_array / frequencies <= 0):
        raise ValueError("each yield must be greater than its negative coupon frequency")
    times, amounts, offsets = flatten_bond_cashflows(items, settlement=settlement)
    selected = _resolve_engine(engine, times.size)
    if selected == "native":
        raw = cast(
            dict[str, object],
            _native.require().price_cashflow_batches_from_yield(
                times, amounts, offsets, yield_array, frequencies
            ),
        )
        prices = np.asarray(raw["prices"], dtype=np.float64)
        macaulay = np.asarray(raw["macaulay_durations"], dtype=np.float64)
        convexity = np.asarray(raw["convexities"], dtype=np.float64)
        dv01 = np.asarray(raw["dv01"], dtype=np.float64)
    else:
        prices, macaulay, convexity, dv01 = _numpy_yield_metrics(
            times, amounts, offsets, yield_array, frequencies
        )
    modified = macaulay / (1.0 + yield_array / frequencies)
    accrued = np.asarray(
        [bond.accrued_interest(settlement) for bond in items], dtype=np.float64
    )
    return BondBatchAnalytics(
        dirty_prices=prices,
        clean_prices=prices - accrued,
        accrued_interest=accrued,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
        dv01=dv01,
        engine=selected,
    )


def _price_one_yield(
    times: FloatArray, amounts: FloatArray, yield_value: float, frequency: int
) -> float:
    base = 1.0 + yield_value / frequency
    if base <= 0:
        return float("inf")
    return float(np.dot(amounts, np.power(base, -frequency * times)))


def _numpy_solve_yields(
    items: tuple[FixedRateBond, ...],
    prices: FloatArray,
    settlement: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[FloatArray, NDArray[np.bool_], Int32Array]:
    yields = np.zeros(len(items), dtype=np.float64)
    converged = np.zeros(len(items), dtype=np.bool_)
    iterations = np.zeros(len(items), dtype=np.int32)
    for index, (bond, target) in enumerate(zip(items, prices, strict=True)):
        times, amounts = bond.cashflows(settlement=settlement)
        if times.size == 0:
            raise ValueError("cannot solve yield for a matured bond")
        lower = -0.999999 * bond.frequency
        upper = 1.0
        upper_price = _price_one_yield(times, amounts, upper, bond.frequency)
        while upper_price > target and upper < 1e6:
            upper = 2.0 * upper + 1.0
            upper_price = _price_one_yield(times, amounts, upper, bond.frequency)
        if upper_price > target:
            yields[index] = upper
            continue
        middle = 0.0
        for iteration in range(1, max_iterations + 1):
            middle = 0.5 * (lower + upper)
            error = _price_one_yield(times, amounts, middle, bond.frequency) - target
            if abs(error) <= tolerance * max(1.0, target) or abs(upper - lower) <= tolerance:
                converged[index] = True
                iterations[index] = iteration
                break
            if error > 0:
                lower = middle
            else:
                upper = middle
        else:
            iterations[index] = max_iterations
        yields[index] = middle
    return yields, converged, iterations


def yield_from_prices(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    prices: ArrayLike,
    *,
    settlement: float = 0.0,
    tolerance: float = 1.0e-12,
    max_iterations: int = 200,
    engine: Engine = "auto",
) -> YieldSolveResult:
    """Solve nominal annual yields using a bounded monotone bisection."""

    items = _as_bond_tuple(bonds)
    targets = _broadcast_values(prices, len(items), "prices")
    if np.any(targets <= 0):
        raise ValueError("prices must be positive")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    try:
        iteration_limit = integer_index(max_iterations)
    except TypeError as exc:
        raise ValueError("max_iterations must be a positive integer") from exc
    if isinstance(max_iterations, bool) or iteration_limit <= 0:
        raise ValueError("max_iterations must be positive")
    times, amounts, offsets = flatten_bond_cashflows(items, settlement=settlement)
    selected = _resolve_engine(engine, max(times.size, len(items) * 64))
    if selected == "native":
        frequencies = np.asarray([bond.frequency for bond in items], dtype=np.int32)
        raw = cast(
            dict[str, object],
            _native.require().solve_yields_from_prices(
                times,
                amounts,
                offsets,
                targets,
                frequencies,
                tolerance,
                iteration_limit,
            ),
        )
        solved = np.asarray(raw["yields"], dtype=np.float64)
        converged = np.asarray(raw["converged"], dtype=np.uint8).astype(np.bool_)
        iterations = np.asarray(raw["iterations"], dtype=np.int32)
    else:
        solved, converged, iterations = _numpy_solve_yields(
            items, targets, settlement, tolerance, iteration_limit
        )
    return YieldSolveResult(solved, converged, iterations, selected)


__all__ = [
    "BondBatchAnalytics",
    "CashFlow",
    "Engine",
    "FixedRateBond",
    "YieldSolveResult",
    "flatten_bond_cashflows",
    "price_bonds",
    "price_bonds_from_yield",
    "yield_from_prices",
]
