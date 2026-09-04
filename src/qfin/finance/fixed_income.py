"""Fixed-rate cash flows, batch valuation, yield solving, and rate risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import floor, isfinite
from operator import index as integer_index
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin import _native
from qfin.finance.curves import YieldCurve
from qfin.finance.dates import (
    BusinessDayConvention,
    Calendar,
    DateLike,
    Schedule,
    as_date,
)
from qfin.finance.daycount import DayCountConvention, year_fraction

FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]
Int32Array = NDArray[np.int32]
Engine = Literal["auto", "numpy", "native"]
Settlement = float | DateLike | None

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


@dataclass(frozen=True, slots=True, init=False)
class FixedRateBond:
    """Fixed-rate bond defined either by year fractions or calendar dates.

    The original ``maturity``-in-years constructor remains unchanged.  Supplying
    ``issue_date`` and ``maturity_date`` activates convention-aware schedules,
    settlement, stubs, business-day adjustment, and dated accrual.
    """

    maturity: float
    coupon_rate: float
    face_value: float = 100.0
    frequency: int = 2
    issue_date: date | None = None
    maturity_date: date | None = None
    day_count: DayCountConvention = DayCountConvention.THIRTY_360
    calendar: Calendar = field(default_factory=Calendar)
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    termination_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    date_generation: Literal["forward", "backward"] = "backward"
    end_of_month: bool = False
    first_coupon_date: date | None = None
    next_to_last_coupon_date: date | None = None
    _dated_schedule: Schedule | None = None

    def __init__(
        self,
        maturity: float | None = None,
        coupon_rate: float | None = None,
        face_value: float = 100.0,
        frequency: int = 2,
        *,
        issue_date: DateLike | None = None,
        maturity_date: DateLike | None = None,
        day_count: DayCountConvention | str = DayCountConvention.THIRTY_360,
        calendar: Calendar | None = None,
        business_day_convention: BusinessDayConvention | str = (
            BusinessDayConvention.MODIFIED_FOLLOWING
        ),
        termination_convention: BusinessDayConvention | str | None = None,
        date_generation: Literal["forward", "backward"] = "backward",
        end_of_month: bool = False,
        first_coupon_date: DateLike | None = None,
        next_to_last_coupon_date: DateLike | None = None,
    ) -> None:
        if coupon_rate is None:
            raise TypeError("coupon_rate is required")
        values = (coupon_rate, face_value)
        if not all(isfinite(value) for value in values):
            raise ValueError("bond inputs must be finite")
        if face_value <= 0:
            raise ValueError("face_value must be positive")
        try:
            normalized_frequency = integer_index(frequency)
        except TypeError as exc:
            raise ValueError("frequency must be an integer in [1, 365]") from exc
        if isinstance(frequency, bool) or normalized_frequency <= 0 or normalized_frequency > 365:
            raise ValueError("frequency must be an integer in [1, 365]")
        has_issue = issue_date is not None
        has_maturity_date = maturity_date is not None
        if has_issue != has_maturity_date:
            raise ValueError("issue_date and maturity_date must be supplied together")
        selected_day_count = DayCountConvention.parse(day_count)
        selected_calendar = calendar or Calendar()
        selected_convention = BusinessDayConvention.parse(business_day_convention)
        selected_termination = BusinessDayConvention.parse(
            termination_convention or selected_convention
        )
        normalized_issue: date | None = None
        normalized_maturity_date: date | None = None
        normalized_first: date | None = None
        normalized_penultimate: date | None = None
        dated_schedule: Schedule | None = None
        if has_issue:
            if maturity is not None:
                raise ValueError("use maturity_date instead of maturity for a dated bond")
            normalized_issue = as_date(cast(DateLike, issue_date), name="issue date")
            normalized_maturity_date = as_date(cast(DateLike, maturity_date), name="maturity date")
            dated_schedule = Schedule(
                normalized_issue,
                normalized_maturity_date,
                normalized_frequency,
                calendar=selected_calendar,
                business_day_convention=selected_convention,
                termination_convention=selected_termination,
                date_generation=date_generation,
                end_of_month=end_of_month,
                first_coupon_date=first_coupon_date,
                next_to_last_coupon_date=next_to_last_coupon_date,
            )
            normalized_first = dated_schedule.first_coupon_date
            normalized_penultimate = dated_schedule.next_to_last_coupon_date
            normalized_maturity = year_fraction(
                normalized_issue, normalized_maturity_date, selected_day_count
            )
        else:
            if maturity is None or not isfinite(maturity) or maturity <= 0:
                raise ValueError("maturity must be finite and positive")
            if first_coupon_date is not None or next_to_last_coupon_date is not None:
                raise ValueError("dated stub boundaries require issue_date and maturity_date")
            normalized_maturity = float(maturity)
        object.__setattr__(self, "maturity", normalized_maturity)
        object.__setattr__(self, "coupon_rate", float(coupon_rate))
        object.__setattr__(self, "face_value", float(face_value))
        object.__setattr__(self, "frequency", normalized_frequency)
        object.__setattr__(self, "issue_date", normalized_issue)
        object.__setattr__(self, "maturity_date", normalized_maturity_date)
        object.__setattr__(self, "day_count", selected_day_count)
        object.__setattr__(self, "calendar", selected_calendar)
        object.__setattr__(self, "business_day_convention", selected_convention)
        object.__setattr__(self, "termination_convention", selected_termination)
        object.__setattr__(self, "date_generation", date_generation)
        object.__setattr__(self, "end_of_month", bool(end_of_month))
        object.__setattr__(self, "first_coupon_date", normalized_first)
        object.__setattr__(self, "next_to_last_coupon_date", normalized_penultimate)
        object.__setattr__(self, "_dated_schedule", dated_schedule)

    @classmethod
    def from_dates(
        cls,
        issue_date: DateLike,
        maturity_date: DateLike,
        coupon_rate: float,
        **kwargs: Any,
    ) -> FixedRateBond:
        """Construct a convention-aware dated bond."""

        return cls(
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            maturity_date=maturity_date,
            **kwargs,
        )

    @property
    def is_dated(self) -> bool:
        """Return whether the bond uses calendar-date cash flows."""

        return self._dated_schedule is not None

    @property
    def schedule(self) -> Schedule:
        """Return the dated schedule, rejecting floating-time bonds explicitly."""

        if self._dated_schedule is None:
            raise ValueError("floating-time bonds do not have a calendar schedule")
        return self._dated_schedule

    @property
    def payment_dates(self) -> tuple[date, ...]:
        """Return adjusted payment dates for a dated bond."""

        return self.schedule.payment_dates

    def payment_schedule(self) -> FloatArray:
        """Return coupon/principal payment times measured from issue."""

        if self.is_dated:
            assert self.issue_date is not None
            return np.asarray(
                [
                    year_fraction(self.issue_date, item, self.day_count)
                    for item in self.payment_dates
                ],
                dtype=np.float64,
            )

        periods = floor(self.maturity * self.frequency + 1.0e-12)
        times = np.arange(1, periods + 1, dtype=np.float64) / self.frequency
        if times.size == 0 or times[-1] < self.maturity - 1.0e-12:
            times = np.append(times, self.maturity)
        elif times[-1] > self.maturity + 1.0e-12:
            times[-1] = self.maturity
        return np.asarray(times, dtype=np.float64)

    def cashflows(
        self,
        *,
        settlement: Settlement = None,
        time_day_count: DayCountConvention | str | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return remaining times and amounts relative to ``settlement``."""

        if self.is_dated:
            if settlement is None:
                assert self.issue_date is not None
                settlement_date = self.issue_date
            elif isinstance(settlement, (float, int)):
                raise TypeError("settlement for a dated bond must be a date")
            else:
                settlement_date = as_date(settlement, name="settlement")
            schedule = self.schedule
            selected_time_day_count = (
                self.day_count
                if time_day_count is None
                else DayCountConvention.parse(time_day_count)
            )
            accruals = np.asarray(
                [
                    year_fraction(start, end, self.day_count)
                    for start, end in zip(
                        schedule.unadjusted_dates[:-1],
                        schedule.unadjusted_dates[1:],
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            amounts = self.face_value * self.coupon_rate * accruals
            amounts[-1] += self.face_value
            remaining = np.asarray(
                [item > settlement_date for item in schedule.payment_dates], dtype=np.bool_
            )
            times = np.asarray(
                [
                    year_fraction(settlement_date, item, selected_time_day_count)
                    for item, include in zip(schedule.payment_dates, remaining, strict=True)
                    if include
                ],
                dtype=np.float64,
            )
            return np.ascontiguousarray(times), np.ascontiguousarray(amounts[remaining])

        numeric_settlement = 0.0 if settlement is None else settlement
        if not isinstance(numeric_settlement, (float, int)):
            raise TypeError("settlement for a floating-time bond must be numeric")
        numeric_settlement = float(numeric_settlement)
        if not isfinite(numeric_settlement) or numeric_settlement < 0:
            raise ValueError("settlement must be finite and non-negative")
        payment_times = self.payment_schedule()
        previous_times = np.concatenate((np.array([0.0]), payment_times[:-1]))
        accrual_fractions = payment_times - previous_times
        amounts = self.face_value * self.coupon_rate * accrual_fractions
        amounts[-1] += self.face_value
        remaining = payment_times > numeric_settlement + 1.0e-12
        relative_times = np.ascontiguousarray(payment_times[remaining] - numeric_settlement)
        remaining_amounts = np.ascontiguousarray(amounts[remaining])
        return relative_times, remaining_amounts

    def accrued_interest(self, settlement: Settlement = None) -> float:
        """Return straight-line coupon accrual from the preceding payment date."""

        if self.is_dated:
            if settlement is None:
                return 0.0
            if isinstance(settlement, (float, int)):
                raise TypeError("settlement for a dated bond must be a date")
            settlement_date = as_date(settlement, name="settlement")
            dated_schedule = self.schedule
            if (
                settlement_date <= dated_schedule.start_date
                or settlement_date >= dated_schedule.end_date
            ):
                return 0.0
            boundaries = dated_schedule.unadjusted_dates
            dated_completed = [item for item in boundaries if item <= settlement_date]
            dated_previous = dated_completed[-1]
            if dated_previous == settlement_date:
                return 0.0
            following = next(item for item in boundaries if item > settlement_date)
            full_coupon = (
                self.face_value
                * self.coupon_rate
                * year_fraction(dated_previous, following, self.day_count)
            )
            full_fraction = year_fraction(dated_previous, following, self.day_count)
            elapsed_fraction = year_fraction(dated_previous, settlement_date, self.day_count)
            if full_fraction <= 0:
                return 0.0
            return full_coupon * elapsed_fraction / full_fraction

        numeric_settlement = 0.0 if settlement is None else settlement
        if not isinstance(numeric_settlement, (float, int)):
            raise TypeError("settlement for a floating-time bond must be numeric")
        numeric_settlement = float(numeric_settlement)
        if not isfinite(numeric_settlement) or numeric_settlement < 0:
            raise ValueError("settlement must be finite and non-negative")
        if numeric_settlement <= 0 or numeric_settlement >= self.maturity:
            return 0.0
        numeric_schedule = self.payment_schedule()
        numeric_completed = numeric_schedule[numeric_schedule <= numeric_settlement + 1.0e-12]
        numeric_previous = 0.0 if numeric_completed.size == 0 else float(numeric_completed[-1])
        if abs(numeric_previous - numeric_settlement) <= 1.0e-12:
            return 0.0
        return self.face_value * self.coupon_rate * (numeric_settlement - numeric_previous)


@dataclass(frozen=True, slots=True)
class BondBatchAnalytics:
    """Vectorized bond valuation with explicit risk methodologies.

    ``macaulay_duration``, ``modified_duration``, and ``convexity`` retain the
    QFin 1.0 compatibility fields.  New code should use the explicitly named
    curve or YTM fields and inspect ``methodology``.
    """

    dirty_prices: FloatArray
    clean_prices: FloatArray
    accrued_interest: FloatArray
    macaulay_duration: FloatArray
    modified_duration: FloatArray
    convexity: FloatArray
    dv01: FloatArray
    engine: Literal["numpy", "native"]
    methodology: Literal["curve_parallel_zero", "yield_to_maturity"]
    parallel_zero_duration: FloatArray
    effective_duration: FloatArray
    effective_convexity: FloatArray
    spread_duration: FloatArray
    cs01: FloatArray
    ytm_macaulay_duration: FloatArray
    ytm_modified_duration: FloatArray
    ytm_convexity: FloatArray

    @property
    def pv01(self) -> FloatArray:
        """Compatibility-neutral name for the one-basis-point price change."""

        return self.dv01


@dataclass(frozen=True, slots=True)
class KeyRateRiskReport:
    """Central-bump risk by zero-curve node for every bond in a batch."""

    node_times: FloatArray
    base_prices: FloatArray
    key_rate_dv01: FloatArray
    key_rate_duration: FloatArray
    parallel_dv01: FloatArray
    bump_size: float
    interpolation: str
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
    settlement: Settlement = None,
    time_day_count: DayCountConvention | str | None = None,
) -> tuple[FloatArray, FloatArray, Int64Array]:
    """Flatten a bond batch into contiguous buffers and prefix offsets."""

    items = _as_bond_tuple(bonds)
    time_parts: list[FloatArray] = []
    amount_parts: list[FloatArray] = []
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    for index, bond in enumerate(items):
        times, amounts = bond.cashflows(
            settlement=settlement,
            time_day_count=time_day_count,
        )
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


def _resolve_engine(
    engine: Engine,
    workload: int,
    *,
    native_compatible: bool = True,
) -> Literal["numpy", "native"]:
    if engine not in ("auto", "numpy", "native"):
        raise ValueError("engine must be 'auto', 'numpy', or 'native'")
    if engine == "native":
        if not native_compatible:
            raise ValueError(
                "native engine requires linear-zero interpolation with flat-zero extrapolation"
            )
        _native.require()
        return "native"
    if engine == "numpy":
        return "numpy"
    if native_compatible and _native.available() and workload >= _AUTO_NATIVE_CASHFLOW_THRESHOLD:
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
    base_discounts = np.asarray(curve.discount(times), dtype=np.float64)
    shifted_discounts = base_discounts * np.exp(-shift * times)
    present_values = amounts * shifted_discounts
    prices = _segment_sum(present_values, offsets)
    first = _segment_sum(times * present_values, offsets)
    second = _segment_sum(times * times * present_values, offsets)
    durations = np.divide(first, prices, out=np.zeros_like(prices), where=prices != 0)
    convexities = np.divide(second, prices, out=np.zeros_like(prices), where=prices != 0)
    down = _segment_sum(amounts * shifted_discounts * np.exp(1.0e-4 * times), offsets)
    up = _segment_sum(amounts * shifted_discounts * np.exp(-1.0e-4 * times), offsets)
    return prices, durations, convexities, 0.5 * (down - up)


def _curve_effective_metrics(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    curve: YieldCurve,
    shift: float,
    bump_size: float = 1.0e-4,
) -> tuple[FloatArray, FloatArray]:
    base_discounts = np.asarray(curve.discount(times), dtype=np.float64)
    shifted_discounts = base_discounts * np.exp(-shift * times)
    prices = _segment_sum(amounts * shifted_discounts, offsets)
    down = _segment_sum(amounts * shifted_discounts * np.exp(bump_size * times), offsets)
    up = _segment_sum(amounts * shifted_discounts * np.exp(-bump_size * times), offsets)
    duration = np.divide(
        down - up,
        2.0 * bump_size * prices,
        out=np.zeros_like(prices),
        where=prices != 0,
    )
    convexity = np.divide(
        down + up - 2.0 * prices,
        bump_size * bump_size * prices,
        out=np.zeros_like(prices),
        where=prices != 0,
    )
    return duration, convexity


def _curve_settlement(
    items: tuple[FixedRateBond, ...],
    curve: YieldCurve,
    settlement: Settlement,
) -> Settlement:
    dated = {bond.is_dated for bond in items}
    if len(dated) > 1:
        raise ValueError("dated and floating-time bonds cannot share a valuation batch")
    if not dated or False in dated:
        return 0.0 if settlement is None else settlement
    if settlement is None:
        if curve.valuation_date is None:
            raise ValueError("dated bond pricing requires settlement or curve valuation_date")
        normalized = curve.valuation_date
    else:
        if isinstance(settlement, (float, int)):
            raise TypeError("settlement for dated bond pricing must be a date")
        normalized = as_date(settlement, name="settlement")
    if curve.valuation_date is not None and normalized != curve.valuation_date:
        raise ValueError("dated bond settlement must equal the curve valuation_date")
    return normalized


def price_bonds(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    curve: YieldCurve,
    *,
    settlement: Settlement = None,
    parallel_shift: float = 0.0,
    z_spread: float = 0.0,
    engine: Engine = "auto",
) -> BondBatchAnalytics:
    """Price bonds against a zero curve and report explicit curve/spread risk.

    ``parallel_shift`` moves every continuously compounded zero rate while
    ``z_spread`` is an additive continuous spread.  For deterministic cash
    flows their local sensitivities are numerically equal, but both labels are
    retained so the methodology is never implicit.
    """

    if not isfinite(parallel_shift) or not isfinite(z_spread):
        raise ValueError("parallel_shift and z_spread must be finite")
    items = _as_bond_tuple(bonds)
    normalized_settlement = _curve_settlement(items, curve, settlement)
    time_day_count = curve.day_count if items and items[0].is_dated else None
    times, amounts, offsets = flatten_bond_cashflows(
        items,
        settlement=normalized_settlement,
        time_day_count=time_day_count,
    )
    total_shift = parallel_shift + z_spread
    selected = _resolve_engine(
        engine,
        times.size,
        native_compatible=curve.native_compatible,
    )
    if selected == "native":
        raw = cast(
            dict[str, object],
            _native.require().price_cashflow_batches(
                times,
                amounts,
                offsets,
                curve.times,
                curve.zero_rates,
                total_shift,
            ),
        )
        prices = np.asarray(raw["prices"], dtype=np.float64)
        durations = np.asarray(raw["macaulay_durations"], dtype=np.float64)
        convexities = np.asarray(raw["convexities"], dtype=np.float64)
        dv01 = np.asarray(raw["dv01"], dtype=np.float64)
    else:
        prices, durations, convexities, dv01 = _numpy_curve_metrics(
            times, amounts, offsets, curve, total_shift
        )
    effective_duration, effective_convexity = _curve_effective_metrics(
        times, amounts, offsets, curve, total_shift
    )
    accrued = np.asarray(
        [bond.accrued_interest(normalized_settlement) for bond in items], dtype=np.float64
    )
    unavailable = np.full(len(items), np.nan, dtype=np.float64)
    return BondBatchAnalytics(
        dirty_prices=prices,
        clean_prices=prices - accrued,
        accrued_interest=accrued,
        macaulay_duration=durations,
        modified_duration=durations.copy(),
        convexity=convexities,
        dv01=dv01,
        engine=selected,
        methodology="curve_parallel_zero",
        parallel_zero_duration=durations.copy(),
        effective_duration=effective_duration,
        effective_convexity=effective_convexity,
        spread_duration=durations.copy(),
        cs01=dv01.copy(),
        ytm_macaulay_duration=unavailable.copy(),
        ytm_modified_duration=unavailable.copy(),
        ytm_convexity=unavailable.copy(),
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
    denominator = prices * frequencies.astype(np.float64) ** 2 * (1.0 + yields / frequencies) ** 2
    convexity = np.divide(numerator, denominator, out=np.zeros_like(prices), where=denominator != 0)

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


def _yield_effective_metrics(
    times: FloatArray,
    amounts: FloatArray,
    offsets: Int64Array,
    yields: FloatArray,
    frequencies: Int32Array,
    bump_size: float = 1.0e-4,
) -> tuple[FloatArray, FloatArray]:
    counts = np.diff(offsets)
    repeated_frequencies = np.repeat(frequencies.astype(np.float64), counts)

    def prices_at(candidate_yields: FloatArray) -> FloatArray:
        repeated = np.repeat(candidate_yields, counts)
        values = amounts * np.power(
            1.0 + repeated / repeated_frequencies,
            -repeated_frequencies * times,
        )
        return _segment_sum(values, offsets)

    prices = prices_at(yields)
    up = prices_at(yields + bump_size)
    central = yields - bump_size > -frequencies
    down_candidates = np.where(central, yields - bump_size, yields)
    down = prices_at(down_candidates)
    duration = np.where(
        central,
        np.divide(
            down - up,
            2.0 * bump_size * prices,
            out=np.zeros_like(prices),
            where=prices != 0,
        ),
        np.divide(
            prices - up,
            bump_size * prices,
            out=np.zeros_like(prices),
            where=prices != 0,
        ),
    )
    convexity = np.full_like(prices, np.nan)
    np.divide(
        down + up - 2.0 * prices,
        bump_size * bump_size * prices,
        out=convexity,
        where=central & (prices != 0),
    )
    return duration, convexity


def price_bonds_from_yield(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    yields: ArrayLike,
    *,
    settlement: Settlement = None,
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
    effective_duration, effective_convexity = _yield_effective_metrics(
        times, amounts, offsets, yield_array, frequencies
    )
    accrued = np.asarray([bond.accrued_interest(settlement) for bond in items], dtype=np.float64)
    unavailable = np.full(len(items), np.nan, dtype=np.float64)
    return BondBatchAnalytics(
        dirty_prices=prices,
        clean_prices=prices - accrued,
        accrued_interest=accrued,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
        dv01=dv01,
        engine=selected,
        methodology="yield_to_maturity",
        parallel_zero_duration=unavailable.copy(),
        effective_duration=effective_duration,
        effective_convexity=effective_convexity,
        spread_duration=unavailable.copy(),
        cs01=unavailable.copy(),
        ytm_macaulay_duration=macaulay.copy(),
        ytm_modified_duration=modified.copy(),
        ytm_convexity=convexity.copy(),
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
    settlement: Settlement,
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
    settlement: Settlement = None,
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


def key_rate_risk(
    bonds: FixedRateBond | list[FixedRateBond] | tuple[FixedRateBond, ...],
    curve: YieldCurve,
    *,
    settlement: Settlement = None,
    bump_size: float = 1.0e-4,
    engine: Engine = "auto",
) -> KeyRateRiskReport:
    """Return central-bump DV01 and duration for every zero-curve node.

    A key-rate shock changes one continuously compounded zero-rate node while
    preserving the curve's interpolation and extrapolation settings.  DV01 is
    normalized to a one-basis-point move even when ``bump_size`` differs.
    """

    if not isfinite(bump_size) or bump_size <= 0:
        raise ValueError("bump_size must be finite and positive")
    items = _as_bond_tuple(bonds)
    base = price_bonds(items, curve, settlement=settlement, engine=engine)
    key_dv01 = np.zeros((len(items), curve.times.size), dtype=np.float64)
    key_duration = np.zeros_like(key_dv01)
    for node in range(curve.times.size):
        shock = np.zeros(curve.times.size, dtype=np.float64)
        shock[node] = bump_size
        up = price_bonds(
            items,
            curve.shifted(shock),
            settlement=settlement,
            engine=engine,
        ).dirty_prices
        down = price_bonds(
            items,
            curve.shifted(-shock),
            settlement=settlement,
            engine=engine,
        ).dirty_prices
        derivative = (down - up) / (2.0 * bump_size)
        key_dv01[:, node] = derivative * 1.0e-4
        key_duration[:, node] = np.divide(
            derivative,
            base.dirty_prices,
            out=np.zeros_like(derivative),
            where=base.dirty_prices != 0,
        )
    parallel_up = price_bonds(
        items,
        curve.shifted(bump_size),
        settlement=settlement,
        engine=engine,
    ).dirty_prices
    parallel_down = price_bonds(
        items,
        curve.shifted(-bump_size),
        settlement=settlement,
        engine=engine,
    ).dirty_prices
    parallel_dv01 = (parallel_down - parallel_up) / (2.0 * bump_size) * 1.0e-4
    return KeyRateRiskReport(
        node_times=curve.times.copy(),
        base_prices=base.dirty_prices.copy(),
        key_rate_dv01=key_dv01,
        key_rate_duration=key_duration,
        parallel_dv01=parallel_dv01,
        bump_size=bump_size,
        interpolation=curve.interpolation.value,
        engine=base.engine,
    )


def _with_coupon_rate(bond: FixedRateBond, coupon_rate: float) -> FixedRateBond:
    if not bond.is_dated:
        return FixedRateBond(
            bond.maturity,
            coupon_rate,
            face_value=bond.face_value,
            frequency=bond.frequency,
        )
    assert bond.issue_date is not None and bond.maturity_date is not None
    return FixedRateBond(
        coupon_rate=coupon_rate,
        face_value=bond.face_value,
        frequency=bond.frequency,
        issue_date=bond.issue_date,
        maturity_date=bond.maturity_date,
        day_count=bond.day_count,
        calendar=bond.calendar,
        business_day_convention=bond.business_day_convention,
        termination_convention=bond.termination_convention,
        date_generation=bond.date_generation,
        end_of_month=bond.end_of_month,
        first_coupon_date=bond.first_coupon_date,
        next_to_last_coupon_date=bond.next_to_last_coupon_date,
    )


def par_yield(
    bond: FixedRateBond,
    curve: YieldCurve,
    *,
    settlement: Settlement = None,
    target_clean_price: float | None = None,
) -> float:
    """Return the coupon rate that prices the bond at the target clean price.

    Cash flows and accrued interest are affine in the coupon rate, so the par
    rate is solved exactly from zero- and unit-coupon valuations.  By default
    the target is the bond's face value.
    """

    target = bond.face_value if target_clean_price is None else target_clean_price
    if not isfinite(target) or target <= 0:
        raise ValueError("target_clean_price must be finite and positive")
    zero = price_bonds(
        _with_coupon_rate(bond, 0.0),
        curve,
        settlement=settlement,
        engine="numpy",
    ).clean_prices[0]
    unit = price_bonds(
        _with_coupon_rate(bond, 1.0),
        curve,
        settlement=settlement,
        engine="numpy",
    ).clean_prices[0]
    coupon_value = float(unit - zero)
    if abs(coupon_value) <= np.finfo(np.float64).eps:
        raise ValueError("par yield is undefined because no coupon cash flows remain")
    return float((target - zero) / coupon_value)


__all__ = [
    "BondBatchAnalytics",
    "CashFlow",
    "Engine",
    "FixedRateBond",
    "KeyRateRiskReport",
    "Settlement",
    "YieldSolveResult",
    "flatten_bond_cashflows",
    "key_rate_risk",
    "par_yield",
    "price_bonds",
    "price_bonds_from_yield",
    "yield_from_prices",
]
