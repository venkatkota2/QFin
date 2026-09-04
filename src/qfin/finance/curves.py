"""Convention-aware yield curves and curve diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from math import isfinite
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from qfin.finance.dates import DateLike, ValuationDate, as_date
from qfin.finance.daycount import DayCountConvention, year_fraction
from qfin.finance.rates import (
    Compounding,
    continuous_rate,
    discount_factor,
    rate_from_discount_factor,
)

FloatArray = NDArray[np.float64]


class CurveInterpolation(StrEnum):
    """Validated term-structure interpolation choices."""

    LINEAR_ZERO = "linear_zero"
    LINEAR_DISCOUNT = "linear_discount"
    LOG_LINEAR_DISCOUNT = "log_linear_discount"
    MONOTONE_ZERO = "monotone_zero"

    @classmethod
    def parse(cls, value: CurveInterpolation | str) -> CurveInterpolation:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "linear": cls.LINEAR_ZERO,
            "linear_zero_rate": cls.LINEAR_ZERO,
            "linear_discount_factor": cls.LINEAR_DISCOUNT,
            "log_linear": cls.LOG_LINEAR_DISCOUNT,
            "loglinear_discount": cls.LOG_LINEAR_DISCOUNT,
            "log_linear_discount_factor": cls.LOG_LINEAR_DISCOUNT,
            "monotone": cls.MONOTONE_ZERO,
            "pchip": cls.MONOTONE_ZERO,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"curve interpolation must be one of: {choices}") from exc


class CurveExtrapolation(StrEnum):
    """Explicit behavior outside the supplied curve nodes."""

    FLAT_ZERO = "flat_zero"
    FLAT_FORWARD = "flat_forward"
    ERROR = "error"

    @classmethod
    def parse(cls, value: CurveExtrapolation | str) -> CurveExtrapolation:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {"flat": cls.FLAT_ZERO, "raise": cls.ERROR}
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"curve extrapolation must be one of: {choices}") from exc


@dataclass(frozen=True, slots=True)
class CurveMarketQuote:
    """A transparent market input used to construct a node curve.

    This is deliberately not a bootstrap instrument.  QFin 1.1 accepts direct
    zero-rate or discount-factor market nodes here; instrument repricing and
    solved bootstrap nodes belong to the separate calibration layer.
    """

    time: float
    value: float
    quote_type: Literal["zero_rate", "discount_factor"] = "zero_rate"
    instrument: str = "market_node"
    identifier: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.time) or self.time < 0:
            raise ValueError("market-quote time must be finite and non-negative")
        if not isfinite(self.value):
            raise ValueError("market-quote value must be finite")
        if self.quote_type not in ("zero_rate", "discount_factor"):
            raise ValueError("quote_type must be 'zero_rate' or 'discount_factor'")
        if self.quote_type == "discount_factor" and self.value <= 0:
            raise ValueError("discount-factor market quotes must be positive")
        if not self.instrument.strip():
            raise ValueError("market-quote instrument must not be empty")


@dataclass(frozen=True, slots=True)
class CurveDiagnostics:
    """Numerical and economic diagnostics for a constructed curve."""

    node_count: int
    minimum_discount_factor: float
    maximum_discount_factor: float
    minimum_forward_rate: float | None
    maximum_forward_rate: float | None
    increasing_discount_intervals: tuple[int, ...]
    negative_forward_intervals: tuple[int, ...]
    warnings: tuple[str, ...]

    @property
    def has_positive_discount_factors(self) -> bool:
        return self.minimum_discount_factor > 0.0


@dataclass(frozen=True, slots=True, init=False)
class YieldCurve:
    """Discount term structure with canonical continuous zero-rate nodes.

    ``zero_rates`` are stored canonically as continuously compounded rates so
    existing QFin native kernels retain their documented semantics.  The
    ``quote_compounding`` field records how constructor inputs were quoted, and
    :meth:`quoted_zero_rate` returns rates in any supported convention.  This
    makes conversions explicit without breaking the QFin 1.0 public API.
    """

    times: FloatArray
    zero_rates: FloatArray
    interpolation: CurveInterpolation
    extrapolation: CurveExtrapolation
    quote_compounding: Compounding
    valuation_date: date | None
    day_count: DayCountConvention
    input_type: str
    market_quotes: tuple[CurveMarketQuote, ...]
    _node_discount_factors: FloatArray = field(repr=False)
    _monotone: object | None = field(repr=False, compare=False)

    def __init__(
        self,
        times: ArrayLike,
        zero_rates: ArrayLike,
        interpolation: CurveInterpolation | str = CurveInterpolation.LINEAR_ZERO,
        extrapolation: CurveExtrapolation | str = CurveExtrapolation.FLAT_ZERO,
        *,
        compounding: Compounding | str = Compounding.CONTINUOUS,
        valuation_date: DateLike | ValuationDate | None = None,
        day_count: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
    ) -> None:
        node_times = np.array(times, dtype=np.float64, order="C", copy=True).reshape(-1)
        quoted_rates = np.array(zero_rates, dtype=np.float64, order="C", copy=True).reshape(-1)
        if node_times.size == 0 or node_times.shape != quoted_rates.shape:
            raise ValueError("times and zero_rates must have equal non-zero length")
        if not np.all(np.isfinite(node_times)) or np.any(node_times < 0):
            raise ValueError("curve times must be finite and non-negative")
        if not np.all(np.isfinite(quoted_rates)):
            raise ValueError("zero rates must be finite")
        if np.any(np.diff(node_times) <= 0):
            raise ValueError("curve times must be strictly increasing")
        selected_compounding = Compounding.parse(compounding)
        canonical_rates = np.asarray(
            [
                continuous_rate(float(rate), selected_compounding, time=float(time))
                for time, rate in zip(node_times, quoted_rates, strict=True)
            ],
            dtype=np.float64,
        )
        node_discounts = np.exp(-canonical_rates * node_times)
        if not np.all(np.isfinite(node_discounts)) or np.any(node_discounts <= 0):
            raise ValueError("zero rates imply non-finite or non-positive discount factors")
        selected_interpolation = CurveInterpolation.parse(interpolation)
        selected_extrapolation = CurveExtrapolation.parse(extrapolation)
        selected_day_count = DayCountConvention.parse(day_count)
        normalized_valuation_date: date | None
        if isinstance(valuation_date, ValuationDate):
            normalized_valuation_date = valuation_date.value
        else:
            normalized_valuation_date = (
                None if valuation_date is None else as_date(valuation_date, name="valuation date")
            )
        monotone: object | None = None
        if selected_interpolation is CurveInterpolation.MONOTONE_ZERO and node_times.size > 1:
            monotone = PchipInterpolator(node_times, canonical_rates, extrapolate=False)
        node_times.setflags(write=False)
        canonical_rates.setflags(write=False)
        node_discounts.setflags(write=False)
        object.__setattr__(self, "times", node_times)
        object.__setattr__(self, "zero_rates", canonical_rates)
        object.__setattr__(self, "interpolation", selected_interpolation)
        object.__setattr__(self, "extrapolation", selected_extrapolation)
        object.__setattr__(self, "quote_compounding", selected_compounding)
        object.__setattr__(self, "valuation_date", normalized_valuation_date)
        object.__setattr__(self, "day_count", selected_day_count)
        object.__setattr__(self, "input_type", "zero_rate")
        object.__setattr__(self, "market_quotes", ())
        object.__setattr__(self, "_node_discount_factors", node_discounts)
        object.__setattr__(self, "_monotone", monotone)

    @classmethod
    def from_discount_factors(
        cls,
        times: ArrayLike,
        discount_factors: ArrayLike,
        *,
        interpolation: CurveInterpolation | str = CurveInterpolation.LOG_LINEAR_DISCOUNT,
        extrapolation: CurveExtrapolation | str = CurveExtrapolation.FLAT_ZERO,
        valuation_date: DateLike | ValuationDate | None = None,
        day_count: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
    ) -> YieldCurve:
        """Construct a term structure directly from positive discount factors."""

        node_times = np.asarray(times, dtype=np.float64).reshape(-1)
        discounts = np.asarray(discount_factors, dtype=np.float64).reshape(-1)
        if node_times.size == 0 or node_times.shape != discounts.shape:
            raise ValueError("times and discount_factors must have equal non-zero length")
        if not np.all(np.isfinite(node_times)) or np.any(node_times < 0):
            raise ValueError("curve times must be finite and non-negative")
        if np.any(np.diff(node_times) <= 0):
            raise ValueError("curve times must be strictly increasing")
        if not np.all(np.isfinite(discounts)) or np.any(discounts <= 0):
            raise ValueError("discount factors must be finite and positive")
        zero_nodes = np.empty_like(discounts)
        positive = node_times > 0
        zero_nodes[positive] = -np.log(discounts[positive]) / node_times[positive]
        if np.any(~positive):
            if not np.isclose(discounts[~positive][0], 1.0, rtol=0.0, atol=1.0e-14):
                raise ValueError("discount factor at time zero must equal one")
            zero_nodes[~positive] = zero_nodes[positive][0] if np.any(positive) else 0.0
        curve = cls(
            node_times,
            zero_nodes,
            interpolation,
            extrapolation,
            compounding=Compounding.CONTINUOUS,
            valuation_date=valuation_date,
            day_count=day_count,
        )
        copied_discounts = np.array(discounts, dtype=np.float64, order="C", copy=True)
        copied_discounts.setflags(write=False)
        object.__setattr__(curve, "input_type", "discount_factor")
        object.__setattr__(curve, "_node_discount_factors", copied_discounts)
        return curve

    @classmethod
    def from_forward_rates(
        cls,
        times: ArrayLike,
        forward_rates: ArrayLike,
        *,
        compounding: Compounding | str = Compounding.CONTINUOUS,
        interpolation: CurveInterpolation | str = CurveInterpolation.LOG_LINEAR_DISCOUNT,
        extrapolation: CurveExtrapolation | str = CurveExtrapolation.FLAT_FORWARD,
        valuation_date: DateLike | ValuationDate | None = None,
        day_count: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
    ) -> YieldCurve:
        """Construct from one forward quote for every adjacent time interval."""

        node_times = np.asarray(times, dtype=np.float64).reshape(-1)
        forwards = np.asarray(forward_rates, dtype=np.float64).reshape(-1)
        if node_times.size < 2 or forwards.shape != (node_times.size - 1,):
            raise ValueError("forward_rates must contain one value per adjacent time interval")
        if node_times[0] != 0.0:
            raise ValueError("forward-rate curve times must start at zero")
        if not np.all(np.isfinite(node_times)) or np.any(np.diff(node_times) <= 0):
            raise ValueError("curve times must be finite and strictly increasing")
        if not np.all(np.isfinite(forwards)):
            raise ValueError("forward rates must be finite")
        selected_compounding = Compounding.parse(compounding)
        discounts = np.ones(node_times.size, dtype=np.float64)
        for position, forward in enumerate(forwards):
            interval = float(node_times[position + 1] - node_times[position])
            discounts[position + 1] = discounts[position] * discount_factor(
                float(forward), interval, selected_compounding
            )
        curve = cls.from_discount_factors(
            node_times,
            discounts,
            interpolation=interpolation,
            extrapolation=extrapolation,
            valuation_date=valuation_date,
            day_count=day_count,
        )
        object.__setattr__(curve, "input_type", "forward_rate")
        object.__setattr__(curve, "quote_compounding", selected_compounding)
        return curve

    @classmethod
    def from_market_quotes(
        cls,
        quotes: list[CurveMarketQuote] | tuple[CurveMarketQuote, ...],
        *,
        compounding: Compounding | str = Compounding.CONTINUOUS,
        interpolation: CurveInterpolation | str = CurveInterpolation.LINEAR_ZERO,
        extrapolation: CurveExtrapolation | str = CurveExtrapolation.FLAT_ZERO,
        valuation_date: DateLike | ValuationDate | None = None,
        day_count: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
    ) -> YieldCurve:
        """Construct from homogeneous direct market-node quotes."""

        items = tuple(quotes)
        if not items or any(not isinstance(item, CurveMarketQuote) for item in items):
            raise ValueError("quotes must contain CurveMarketQuote objects")
        quote_types = {item.quote_type for item in items}
        if len(quote_types) != 1:
            raise ValueError("direct market-node quote types must be homogeneous")
        times = [item.time for item in items]
        values = [item.value for item in items]
        if items[0].quote_type == "discount_factor":
            curve = cls.from_discount_factors(
                times,
                values,
                interpolation=interpolation,
                extrapolation=extrapolation,
                valuation_date=valuation_date,
                day_count=day_count,
            )
        else:
            curve = cls(
                times,
                values,
                interpolation,
                extrapolation,
                compounding=compounding,
                valuation_date=valuation_date,
                day_count=day_count,
            )
        object.__setattr__(curve, "input_type", "market_quote")
        object.__setattr__(curve, "market_quotes", items)
        return curve

    @property
    def node_discount_factors(self) -> FloatArray:
        """Immutable discount factors at the supplied nodes."""

        return self._node_discount_factors

    @property
    def native_compatible(self) -> bool:
        """Whether current native kernels reproduce this interpolation exactly."""

        return self.interpolation is CurveInterpolation.LINEAR_ZERO and self.extrapolation is (
            CurveExtrapolation.FLAT_ZERO
        )

    def _validate_query(self, time: float | ArrayLike) -> FloatArray:
        query = np.asarray(time, dtype=np.float64)
        if not np.all(np.isfinite(query)) or np.any(query < 0):
            raise ValueError("query times must be finite and non-negative")
        if self.extrapolation is CurveExtrapolation.ERROR and (
            np.any(query < self.times[0]) or np.any(query > self.times[-1])
        ):
            raise ValueError("query time is outside the curve domain")
        return query

    def _flat_forward_log_discount(self, query: FloatArray, *, left: bool) -> FloatArray:
        if self.times.size == 1:
            return np.asarray(-self.zero_rates[0] * query, dtype=np.float64)
        position = 0 if left else self.times.size - 2
        start_time = self.times[position]
        end_time = self.times[position + 1]
        start_log = np.log(self._node_discount_factors[position])
        end_log = np.log(self._node_discount_factors[position + 1])
        slope = (end_log - start_log) / (end_time - start_time)
        anchor_time = start_time if left else end_time
        anchor_log = start_log if left else end_log
        return np.asarray(anchor_log + slope * (query - anchor_time), dtype=np.float64)

    def _discount_array(self, query: FloatArray) -> FloatArray:
        if self.interpolation is CurveInterpolation.LINEAR_ZERO:
            rates = np.interp(query, self.times, self.zero_rates)
            result = np.exp(-rates * query)
        elif self.interpolation is CurveInterpolation.MONOTONE_ZERO:
            if self._monotone is None:
                rates = np.full_like(query, self.zero_rates[0], dtype=np.float64)
            else:
                clipped = np.clip(query, self.times[0], self.times[-1])
                rates = np.asarray(self._monotone(clipped), dtype=np.float64)  # type: ignore[operator]
            result = np.exp(-rates * query)
        elif self.interpolation is CurveInterpolation.LINEAR_DISCOUNT:
            result = np.interp(query, self.times, self._node_discount_factors)
        else:
            log_discounts = np.log(self._node_discount_factors)
            result = np.exp(np.interp(query, self.times, log_discounts))

        before = query < self.times[0]
        after = query > self.times[-1]
        if self.extrapolation is CurveExtrapolation.FLAT_ZERO:
            result = np.where(before, np.exp(-self.zero_rates[0] * query), result)
            result = np.where(after, np.exp(-self.zero_rates[-1] * query), result)
        elif self.extrapolation is CurveExtrapolation.FLAT_FORWARD:
            result = np.where(
                before,
                np.exp(self._flat_forward_log_discount(query, left=True)),
                result,
            )
            result = np.where(
                after,
                np.exp(self._flat_forward_log_discount(query, left=False)),
                result,
            )
        result = np.where(query == 0.0, 1.0, result)
        return np.asarray(result, dtype=np.float64)

    @overload
    def zero_rate(self, time: float) -> float: ...

    @overload
    def zero_rate(self, time: ArrayLike) -> FloatArray: ...

    def zero_rate(self, time: float | ArrayLike) -> float | FloatArray:
        """Return canonical continuously compounded zero rates."""

        query = self._validate_query(time)
        discounts = self._discount_array(query)
        safe_time = np.where(query == 0.0, 1.0, query)
        rates = -np.log(discounts) / safe_time
        zero_value = float(np.interp(0.0, self.times, self.zero_rates))
        rates = np.where(query == 0.0, zero_value, rates)
        if query.ndim == 0:
            return float(rates)
        return np.asarray(rates, dtype=np.float64)

    @overload
    def quoted_zero_rate(
        self,
        time: float,
        compounding: Compounding | str | None = None,
    ) -> float: ...

    @overload
    def quoted_zero_rate(
        self,
        time: ArrayLike,
        compounding: Compounding | str | None = None,
    ) -> FloatArray: ...

    def quoted_zero_rate(
        self,
        time: float | ArrayLike,
        compounding: Compounding | str | None = None,
    ) -> float | FloatArray:
        """Return zero rates in an explicitly selected quote convention."""

        query = self._validate_query(time)
        selected = self.quote_compounding if compounding is None else Compounding.parse(compounding)
        canonical = np.asarray(self.zero_rate(query), dtype=np.float64)
        discounts = self._discount_array(query)
        result = np.empty_like(query, dtype=np.float64)
        flat_query = query.reshape(-1)
        flat_result = result.reshape(-1)
        flat_discounts = discounts.reshape(-1)
        flat_canonical = canonical.reshape(-1)
        for position, value in enumerate(flat_query):
            if value != 0.0:
                flat_result[position] = rate_from_discount_factor(
                    float(flat_discounts[position]), float(value), selected
                )
            elif selected in (Compounding.CONTINUOUS, Compounding.SIMPLE):
                flat_result[position] = float(flat_canonical[position])
            else:
                one_year_discount = float(np.exp(-flat_canonical[position]))
                flat_result[position] = rate_from_discount_factor(
                    one_year_discount, 1.0, selected
                )
        if query.ndim == 0:
            return float(result)
        return result

    @overload
    def discount(self, time: float) -> float: ...

    @overload
    def discount(self, time: ArrayLike) -> FloatArray: ...

    def discount(self, time: float | ArrayLike) -> float | FloatArray:
        """Return discount factors for one or many non-negative times."""

        query = self._validate_query(time)
        discounts = self._discount_array(query)
        if query.ndim == 0:
            return float(discounts)
        return discounts

    def forward_rate(self, start: float, end: float) -> float:
        """Return the continuously compounded forward rate on ``[start, end]``."""

        if not (isfinite(start) and isfinite(end)) or start < 0 or end <= start:
            raise ValueError("require finite times with 0 <= start < end")
        start_discount = self.discount(start)
        end_discount = self.discount(end)
        return -float(np.log(end_discount / start_discount)) / (end - start)

    def time_from_date(self, value: DateLike) -> float:
        """Convert a date to curve time using the curve's day-count metadata."""

        if self.valuation_date is None:
            raise ValueError("curve has no valuation_date")
        return year_fraction(self.valuation_date, value, self.day_count)

    def discount_date(self, value: DateLike) -> float:
        """Return the discount factor for a calendar date."""

        return self.discount(self.time_from_date(value))

    def shifted(self, shift: float | ArrayLike) -> YieldCurve:
        """Return a curve with additive continuous node-rate shocks."""

        shifts = np.asarray(shift, dtype=np.float64)
        if not np.all(np.isfinite(shifts)):
            raise ValueError("curve shifts must be finite")
        if shifts.ndim > 1 or (shifts.ndim == 1 and shifts.shape != self.zero_rates.shape):
            raise ValueError("shift must be scalar or have one value per curve node")
        shifted_curve = YieldCurve(
            self.times,
            np.asarray(self.zero_rates + shifts, dtype=np.float64),
            self.interpolation,
            self.extrapolation,
            compounding=Compounding.CONTINUOUS,
            valuation_date=self.valuation_date,
            day_count=self.day_count,
        )
        object.__setattr__(shifted_curve, "input_type", "shifted_zero_rate")
        return shifted_curve

    def diagnostics(self) -> CurveDiagnostics:
        """Inspect positivity, monotonicity, and adjacent forward rates."""

        discounts = self._node_discount_factors
        if self.times.size > 1:
            forwards = -np.diff(np.log(discounts)) / np.diff(self.times)
            increasing = tuple(int(item) for item in np.flatnonzero(np.diff(discounts) > 0.0))
            negative = tuple(int(item) for item in np.flatnonzero(forwards < 0.0))
            minimum_forward = float(np.min(forwards))
            maximum_forward = float(np.max(forwards))
        else:
            increasing = ()
            negative = ()
            minimum_forward = None
            maximum_forward = None
        warnings: list[str] = []
        if increasing:
            warnings.append("discount factors increase on one or more node intervals")
        if negative:
            warnings.append("one or more adjacent continuously compounded forwards are negative")
        return CurveDiagnostics(
            node_count=int(self.times.size),
            minimum_discount_factor=float(np.min(discounts)),
            maximum_discount_factor=float(np.max(discounts)),
            minimum_forward_rate=minimum_forward,
            maximum_forward_rate=maximum_forward,
            increasing_discount_intervals=increasing,
            negative_forward_intervals=negative,
            warnings=tuple(warnings),
        )

    def explain(self) -> dict[str, object]:
        """Return inspectable construction and convention metadata."""

        return {
            "input_type": self.input_type,
            "quote_compounding": self.quote_compounding.value,
            "canonical_rate_compounding": Compounding.CONTINUOUS.value,
            "interpolation": self.interpolation.value,
            "extrapolation": self.extrapolation.value,
            "valuation_date": (
                None if self.valuation_date is None else self.valuation_date.isoformat()
            ),
            "day_count": self.day_count.value,
            "node_count": int(self.times.size),
            "native_compatible": self.native_compatible,
        }


__all__ = [
    "CurveDiagnostics",
    "CurveExtrapolation",
    "CurveInterpolation",
    "CurveMarketQuote",
    "YieldCurve",
]
