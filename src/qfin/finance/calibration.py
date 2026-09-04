"""Instrument curve bootstrapping with residual-level diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from math import isfinite
from operator import index as integer_index
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from qfin.finance.curves import CurveExtrapolation, CurveInterpolation, YieldCurve
from qfin.finance.dates import (
    BusinessDayConvention,
    Calendar,
    DateLike,
    Schedule,
    as_date,
)
from qfin.finance.daycount import DayCountConvention, year_fraction
from qfin.finance.fixed_income import FixedRateBond, Settlement, price_bonds
from qfin.finance.rates import Compounding, discount_factor, rate_from_discount_factor

FloatArray = NDArray[np.float64]
Maturity: TypeAlias = float | DateLike


def _normalize_maturity(value: Maturity, name: str = "maturity") -> float | date:
    if isinstance(value, (float, int)):
        result = float(value)
        if not isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result
    return as_date(value, name=name)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True, init=False)
class Deposit:
    """Spot-starting deposit quoted as an annualized rate."""

    maturity: float | date
    rate: float
    day_count: DayCountConvention
    compounding: Compounding
    identifier: str

    def __init__(
        self,
        maturity: Maturity,
        rate: float,
        *,
        day_count: DayCountConvention | str = DayCountConvention.ACT_360,
        compounding: Compounding | str = Compounding.SIMPLE,
        identifier: str | None = None,
    ) -> None:
        normalized = _normalize_maturity(maturity)
        object.__setattr__(self, "maturity", normalized)
        object.__setattr__(self, "rate", _finite(rate, "deposit rate"))
        object.__setattr__(self, "day_count", DayCountConvention.parse(day_count))
        object.__setattr__(self, "compounding", Compounding.parse(compounding))
        object.__setattr__(self, "identifier", identifier or f"deposit-{normalized}")


@dataclass(frozen=True, slots=True, init=False)
class ZeroCouponInstrument:
    """Zero-coupon instrument quoted by price per ``face_value``."""

    maturity: float | date
    price: float
    face_value: float
    identifier: str

    def __init__(
        self,
        maturity: Maturity,
        price: float,
        *,
        face_value: float = 100.0,
        identifier: str | None = None,
    ) -> None:
        normalized = _normalize_maturity(maturity)
        normalized_price = _finite(price, "zero-coupon price")
        normalized_face = _finite(face_value, "face_value")
        if normalized_price <= 0 or normalized_face <= 0:
            raise ValueError("zero-coupon price and face_value must be positive")
        object.__setattr__(self, "maturity", normalized)
        object.__setattr__(self, "price", normalized_price)
        object.__setattr__(self, "face_value", normalized_face)
        object.__setattr__(self, "identifier", identifier or f"zero-{normalized}")


@dataclass(frozen=True, slots=True)
class BondMarketQuote:
    """Fixed-rate bond quoted by clean price."""

    bond: FixedRateBond
    clean_price: float
    settlement: Settlement = None
    identifier: str = "bond"

    def __post_init__(self) -> None:
        if not isinstance(self.bond, FixedRateBond):
            raise TypeError("bond must be a FixedRateBond")
        if not isfinite(self.clean_price) or self.clean_price <= 0:
            raise ValueError("clean_price must be finite and positive")
        if not self.identifier.strip():
            raise ValueError("identifier must not be empty")

    @property
    def maturity(self) -> float | date:
        if self.bond.maturity_date is not None:
            return self.bond.maturity_date
        return self.bond.maturity


@dataclass(frozen=True, slots=True, init=False)
class SimpleSwap:
    """Spot-starting, single-curve fixed-for-floating par swap quote."""

    maturity: float | date
    fixed_rate: float
    frequency: int
    fixed_day_count: DayCountConvention
    calendar: Calendar
    business_day_convention: BusinessDayConvention
    end_of_month: bool
    identifier: str

    def __init__(
        self,
        maturity: Maturity,
        fixed_rate: float,
        *,
        frequency: int = 2,
        fixed_day_count: DayCountConvention | str = DayCountConvention.THIRTY_360,
        calendar: Calendar | None = None,
        business_day_convention: BusinessDayConvention | str = (
            BusinessDayConvention.MODIFIED_FOLLOWING
        ),
        end_of_month: bool = False,
        identifier: str | None = None,
    ) -> None:
        normalized = _normalize_maturity(maturity)
        try:
            normalized_frequency = integer_index(frequency)
        except TypeError as exc:
            raise ValueError("swap frequency must divide 12") from exc
        if (
            isinstance(frequency, bool)
            or normalized_frequency <= 0
            or 12 % normalized_frequency != 0
        ):
            raise ValueError("swap frequency must be one of 1, 2, 3, 4, 6, or 12")
        object.__setattr__(self, "maturity", normalized)
        object.__setattr__(self, "fixed_rate", _finite(fixed_rate, "fixed rate"))
        object.__setattr__(self, "frequency", normalized_frequency)
        object.__setattr__(self, "fixed_day_count", DayCountConvention.parse(fixed_day_count))
        object.__setattr__(self, "calendar", calendar or Calendar())
        object.__setattr__(
            self,
            "business_day_convention",
            BusinessDayConvention.parse(business_day_convention),
        )
        object.__setattr__(self, "end_of_month", bool(end_of_month))
        object.__setattr__(self, "identifier", identifier or f"swap-{normalized}")


BootstrapInstrument: TypeAlias = Deposit | ZeroCouponInstrument | BondMarketQuote | SimpleSwap


@dataclass(frozen=True, slots=True)
class BootstrapInstrumentResult:
    """Input/model quote comparison for one bootstrap instrument."""

    identifier: str
    instrument_type: Literal["deposit", "zero_coupon", "bond", "swap"]
    maturity_time: float
    input_quote: float
    model_quote: float
    residual: float
    tolerance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class CurveBootstrapReport:
    """Solved curve nodes and strict instrument-repricing diagnostics."""

    curve: YieldCurve
    input_instruments: tuple[BootstrapInstrument, ...]
    instruments: tuple[BootstrapInstrumentResult, ...]
    node_times: FloatArray
    discount_factors: FloatArray
    zero_rates: FloatArray
    forward_rates: FloatArray
    tolerance: float
    success: bool

    @property
    def maximum_absolute_residual(self) -> float:
        return max((abs(item.residual) for item in self.instruments), default=0.0)

    def explain(self) -> dict[str, object]:
        """Return serializable calibration metadata and diagnostics."""

        return {
            "success": self.success,
            "tolerance": self.tolerance,
            "maximum_absolute_residual": self.maximum_absolute_residual,
            "interpolation": self.curve.interpolation.value,
            "extrapolation": self.curve.extrapolation.value,
            "valuation_date": (
                None if self.curve.valuation_date is None else self.curve.valuation_date.isoformat()
            ),
            "instrument_count": len(self.instruments),
            "instrument_identifiers": tuple(
                item.identifier for item in self.input_instruments
            ),
        }


class CurveBootstrapError(ValueError):
    """Raised when a bootstrap root cannot be found or repricing fails."""


def _maturity_time(
    maturity: float | date,
    valuation_date: date | None,
    curve_day_count: DayCountConvention,
) -> float:
    if isinstance(maturity, float):
        return maturity
    if valuation_date is None:
        raise ValueError("dated bootstrap instruments require valuation_date")
    result = year_fraction(valuation_date, maturity, curve_day_count)
    if result <= 0:
        raise ValueError("bootstrap instrument maturity must follow valuation_date")
    return result


def _instrument_time(
    instrument: BootstrapInstrument,
    valuation_date: date | None,
    curve_day_count: DayCountConvention,
) -> float:
    if isinstance(instrument, BondMarketQuote) and instrument.bond.is_dated:
        return _maturity_time(
            instrument.bond.payment_dates[-1], valuation_date, curve_day_count
        )
    if isinstance(instrument, SimpleSwap) and isinstance(instrument.maturity, date):
        times, _ = _swap_cashflows(instrument, valuation_date, curve_day_count)
        return float(times[-1])
    return _maturity_time(instrument.maturity, valuation_date, curve_day_count)


def _deposit_accrual(instrument: Deposit, valuation_date: date | None) -> float:
    if isinstance(instrument.maturity, float):
        return instrument.maturity
    if valuation_date is None:
        raise ValueError("dated deposits require valuation_date")
    return year_fraction(valuation_date, instrument.maturity, instrument.day_count)


def _swap_cashflows(
    instrument: SimpleSwap,
    valuation_date: date | None,
    curve_day_count: DayCountConvention,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(instrument.maturity, float):
        periods = int(np.floor(instrument.maturity * instrument.frequency + 1.0e-12))
        payment_times = np.arange(1, periods + 1, dtype=np.float64) / instrument.frequency
        if payment_times.size == 0 or payment_times[-1] < instrument.maturity - 1.0e-12:
            payment_times = np.append(payment_times, instrument.maturity)
        previous = np.concatenate((np.asarray([0.0]), payment_times[:-1]))
        return payment_times, payment_times - previous
    if valuation_date is None:
        raise ValueError("dated swaps require valuation_date")
    schedule = Schedule(
        valuation_date,
        instrument.maturity,
        instrument.frequency,
        calendar=instrument.calendar,
        business_day_convention=instrument.business_day_convention,
        end_of_month=instrument.end_of_month,
    )
    times = np.asarray(
        [year_fraction(valuation_date, item, curve_day_count) for item in schedule.payment_dates],
        dtype=np.float64,
    )
    accruals = np.asarray(
        [
            year_fraction(start, end, instrument.fixed_day_count)
            for start, end in zip(
                schedule.unadjusted_dates[:-1], schedule.unadjusted_dates[1:], strict=True
            )
        ],
        dtype=np.float64,
    )
    return times, accruals


def _temporary_curve(
    times: list[float],
    discounts: list[float],
    interpolation: CurveInterpolation,
    extrapolation: CurveExtrapolation,
    valuation_date: date | None,
    day_count: DayCountConvention,
) -> YieldCurve:
    return YieldCurve.from_discount_factors(
        times,
        discounts,
        interpolation=interpolation,
        extrapolation=extrapolation,
        valuation_date=valuation_date,
        day_count=day_count,
    )


def _model_quote(
    instrument: BootstrapInstrument,
    curve: YieldCurve,
    valuation_date: date | None,
) -> float:
    maturity_time = _instrument_time(instrument, valuation_date, curve.day_count)
    if isinstance(instrument, Deposit):
        accrual = _deposit_accrual(instrument, valuation_date)
        maturity_discount = float(curve.discount(maturity_time))
        return rate_from_discount_factor(
            maturity_discount,
            accrual,
            instrument.compounding,
        )
    if isinstance(instrument, ZeroCouponInstrument):
        return float(curve.discount(maturity_time)) * instrument.face_value
    if isinstance(instrument, BondMarketQuote):
        settlement = instrument.settlement
        if settlement is None and instrument.bond.is_dated:
            settlement = valuation_date
        return float(
            price_bonds(
                instrument.bond,
                curve,
                settlement=settlement,
                engine="numpy",
            ).clean_prices[0]
        )
    times, accruals = _swap_cashflows(instrument, valuation_date, curve.day_count)
    discounts = np.asarray(curve.discount(times), dtype=np.float64)
    annuity = float(np.dot(accruals, discounts))
    if annuity <= 0:
        raise CurveBootstrapError(f"{instrument.identifier}: non-positive swap annuity")
    return (1.0 - float(discounts[-1])) / annuity


def _input_quote(instrument: BootstrapInstrument) -> float:
    if isinstance(instrument, Deposit):
        return instrument.rate
    if isinstance(instrument, ZeroCouponInstrument):
        return instrument.price
    if isinstance(instrument, BondMarketQuote):
        return instrument.clean_price
    return instrument.fixed_rate


def _instrument_type(
    instrument: BootstrapInstrument,
) -> Literal["deposit", "zero_coupon", "bond", "swap"]:
    if isinstance(instrument, Deposit):
        return "deposit"
    if isinstance(instrument, ZeroCouponInstrument):
        return "zero_coupon"
    if isinstance(instrument, BondMarketQuote):
        return "bond"
    return "swap"


def _direct_discount(
    instrument: BootstrapInstrument,
    valuation_date: date | None,
) -> float | None:
    if isinstance(instrument, Deposit):
        return discount_factor(
            instrument.rate,
            _deposit_accrual(instrument, valuation_date),
            instrument.compounding,
        )
    if isinstance(instrument, ZeroCouponInstrument):
        return instrument.price / instrument.face_value
    return None


def bootstrap_curve(
    instruments: list[BootstrapInstrument] | tuple[BootstrapInstrument, ...],
    *,
    valuation_date: DateLike | None = None,
    day_count: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
    interpolation: CurveInterpolation | str = CurveInterpolation.LOG_LINEAR_DISCOUNT,
    extrapolation: CurveExtrapolation | str = CurveExtrapolation.FLAT_ZERO,
    tolerance: float = 1.0e-10,
) -> CurveBootstrapReport:
    """Bootstrap a single discount curve and strictly reprice every input.

    Instruments are solved in maturity order.  Deposits and zero-coupon
    instruments imply discount factors directly; bonds and swaps solve the new
    terminal discount factor with a bounded Brent root.  Failure to bracket a
    positive discount factor or residuals above ``tolerance`` raises
    :class:`CurveBootstrapError`.
    """

    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    items = tuple(instruments)
    if not items:
        raise ValueError("at least one bootstrap instrument is required")
    if any(
        not isinstance(item, (Deposit, ZeroCouponInstrument, BondMarketQuote, SimpleSwap))
        for item in items
    ):
        raise TypeError("unsupported bootstrap instrument")
    selected_day_count = DayCountConvention.parse(day_count)
    selected_interpolation = CurveInterpolation.parse(interpolation)
    selected_extrapolation = CurveExtrapolation.parse(extrapolation)
    normalized_valuation = (
        None if valuation_date is None else as_date(valuation_date, name="valuation date")
    )
    ordered = tuple(
        sorted(
            items,
            key=lambda item: _instrument_time(item, normalized_valuation, selected_day_count),
        )
    )
    maturities = [
        _instrument_time(item, normalized_valuation, selected_day_count) for item in ordered
    ]
    if any(right - left <= 1.0e-12 for left, right in pairwise(maturities)):
        raise ValueError("bootstrap instrument maturities must be unique")

    node_times = [0.0]
    discounts = [1.0]
    for instrument, maturity_time in zip(ordered, maturities, strict=True):
        direct = _direct_discount(instrument, normalized_valuation)
        if direct is not None:
            if not isfinite(direct) or direct <= 0:
                raise CurveBootstrapError(
                    f"{instrument.identifier}: implied discount factor is not positive"
                )
            solved_discount = direct
        else:

            def quote_error(
                candidate: float,
                instrument_to_price: BootstrapInstrument = instrument,
                node_maturity: float = maturity_time,
            ) -> float:
                trial_curve = _temporary_curve(
                    [*node_times, node_maturity],
                    [*discounts, candidate],
                    selected_interpolation,
                    selected_extrapolation,
                    normalized_valuation,
                    selected_day_count,
                )
                return _model_quote(
                    instrument_to_price, trial_curve, normalized_valuation
                ) - _input_quote(instrument_to_price)

            lower, upper = 1.0e-8, 5.0
            lower_value = quote_error(lower)
            upper_value = quote_error(upper)
            if lower_value * upper_value > 0:
                raise CurveBootstrapError(
                    f"{instrument.identifier}: positive discount-factor root is not bracketed; "
                    f"residuals=({lower_value:.6g}, {upper_value:.6g})"
                )
            solved_discount = float(
                brentq(quote_error, lower, upper, xtol=min(tolerance * 0.1, 1.0e-13))
            )
        node_times.append(maturity_time)
        discounts.append(solved_discount)

    curve = _temporary_curve(
        node_times,
        discounts,
        selected_interpolation,
        selected_extrapolation,
        normalized_valuation,
        selected_day_count,
    )
    object.__setattr__(curve, "input_type", "bootstrapped_instruments")
    results: list[BootstrapInstrumentResult] = []
    for instrument, maturity_time in zip(ordered, maturities, strict=True):
        input_quote = _input_quote(instrument)
        model_quote = _model_quote(instrument, curve, normalized_valuation)
        residual_value = model_quote - input_quote
        results.append(
            BootstrapInstrumentResult(
                identifier=instrument.identifier,
                instrument_type=_instrument_type(instrument),
                maturity_time=maturity_time,
                input_quote=input_quote,
                model_quote=model_quote,
                residual=residual_value,
                tolerance=tolerance,
                passed=abs(residual_value) <= tolerance,
            )
        )
    failed = [item for item in results if not item.passed]
    if failed:
        detail = ", ".join(f"{item.identifier}={item.residual:.6g}" for item in failed)
        raise CurveBootstrapError(f"instrument repricing exceeded tolerance: {detail}")
    node_array = np.asarray(node_times, dtype=np.float64)
    discount_array = np.asarray(discounts, dtype=np.float64)
    forwards = -np.diff(np.log(discount_array)) / np.diff(node_array)
    return CurveBootstrapReport(
        curve=curve,
        input_instruments=ordered,
        instruments=tuple(results),
        node_times=node_array,
        discount_factors=discount_array,
        zero_rates=curve.zero_rates.copy(),
        forward_rates=forwards,
        tolerance=tolerance,
        success=True,
    )


__all__ = [
    "BondMarketQuote",
    "BootstrapInstrument",
    "BootstrapInstrumentResult",
    "CurveBootstrapError",
    "CurveBootstrapReport",
    "Deposit",
    "SimpleSwap",
    "ZeroCouponInstrument",
    "bootstrap_curve",
]
