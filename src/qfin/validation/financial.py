"""Independent fixed-income references and financial-unit validation reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib import import_module
from importlib.util import find_spec
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.finance.dates import BusinessDayConvention, DateLike, as_date
from qfin.finance.daycount import DayCountConvention
from qfin.finance.fixed_income import FixedRateBond, Settlement

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FinancialTolerance:
    """Absolute, relative, and business-scale acceptance thresholds."""

    absolute: float = 1.0e-10
    relative: float = 1.0e-10
    financial: float = 0.0
    unit: str = "currency units"

    def __post_init__(self) -> None:
        values = (self.absolute, self.relative, self.financial)
        if not all(isfinite(item) and item >= 0 for item in values):
            raise ValueError("validation tolerances must be finite and non-negative")
        if not self.unit.strip():
            raise ValueError("validation tolerance unit must not be empty")

    def allowed_error(self, expected: float) -> float:
        """Return the largest applicable error allowance."""

        return max(self.absolute, self.financial, self.relative * abs(expected))


@dataclass(frozen=True, slots=True)
class FinancialValidationCheck:
    """One actual/reference comparison with an auditable diagnostic."""

    label: str
    actual: float
    expected: float
    difference: float
    relative_difference: float
    allowed_error: float
    unit: str
    passed: bool

    @property
    def diagnostic(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status} {self.label}: actual={self.actual:.12g}, "
            f"expected={self.expected:.12g}, difference={self.difference:.6g} "
            f"{self.unit}, allowed={self.allowed_error:.6g} {self.unit}"
        )


@dataclass(frozen=True, slots=True)
class FinancialValidationReport:
    """Collection of financially meaningful validation checks."""

    name: str
    checks: tuple[FinancialValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.checks)

    @property
    def failed_checks(self) -> tuple[FinancialValidationCheck, ...]:
        return tuple(item for item in self.checks if not item.passed)

    @property
    def maximum_absolute_difference(self) -> float:
        return max((abs(item.difference) for item in self.checks), default=0.0)

    def explain(self) -> dict[str, object]:
        """Return serializable summary metadata."""

        return {
            "name": self.name,
            "passed": self.passed,
            "check_count": len(self.checks),
            "failed_count": len(self.failed_checks),
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "diagnostics": tuple(item.diagnostic for item in self.checks),
        }

    def assert_valid(self) -> None:
        """Raise with detailed financial diagnostics when any check fails."""

        if not self.passed:
            detail = "; ".join(item.diagnostic for item in self.failed_checks)
            raise FinancialValidationError(f"{self.name} validation failed: {detail}")


class FinancialValidationError(AssertionError):
    """Raised by :meth:`FinancialValidationReport.assert_valid`."""


def validate_financial_values(
    name: str,
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    labels: tuple[str, ...] | list[str] | None = None,
    tolerance: FinancialTolerance | None = None,
) -> FinancialValidationReport:
    """Compare vectors using numerical and explicit financial-unit tolerances."""

    selected_tolerance = tolerance or FinancialTolerance()
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected_values = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual_values.shape != expected_values.shape:
        raise ValueError("actual and expected values must have equal shapes")
    if not np.all(np.isfinite(actual_values)) or not np.all(np.isfinite(expected_values)):
        raise ValueError("actual and expected values must be finite")
    normalized_labels = (
        tuple(f"value[{index}]" for index in range(actual_values.size))
        if labels is None
        else tuple(labels)
    )
    if len(normalized_labels) != actual_values.size:
        raise ValueError("labels must contain one entry per value")
    checks: list[FinancialValidationCheck] = []
    for label, actual_value, expected_value in zip(
        normalized_labels, actual_values, expected_values, strict=True
    ):
        difference = float(actual_value - expected_value)
        scale = abs(float(expected_value))
        relative_difference = abs(difference) / scale if scale else abs(difference)
        allowed = selected_tolerance.allowed_error(float(expected_value))
        checks.append(
            FinancialValidationCheck(
                label=label,
                actual=float(actual_value),
                expected=float(expected_value),
                difference=difference,
                relative_difference=relative_difference,
                allowed_error=allowed,
                unit=selected_tolerance.unit,
                passed=abs(difference) <= allowed,
            )
        )
    return FinancialValidationReport(name=name, checks=tuple(checks))


@dataclass(frozen=True, slots=True)
class ReferenceBondAnalytics:
    """Independent nominal-YTM valuation outputs for one bond."""

    dirty_price: float
    clean_price: float
    accrued_interest: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float


def reference_bond_from_yield(
    bond: FixedRateBond,
    yield_rate: float,
    *,
    settlement: Settlement = None,
    bump_size: float = 1.0e-4,
) -> ReferenceBondAnalytics:
    """Value one bond with a scalar reference independent of batch kernels."""

    if not isfinite(yield_rate) or 1.0 + yield_rate / bond.frequency <= 0:
        raise ValueError("yield_rate is outside the nominal-compounding domain")
    if not isfinite(bump_size) or bump_size <= 0:
        raise ValueError("bump_size must be finite and positive")
    times, amounts = bond.cashflows(settlement=settlement)
    if times.size == 0:
        raise ValueError("cannot value a matured bond")

    def scalar_price(candidate: float) -> float:
        base = 1.0 + candidate / bond.frequency
        if base <= 0:
            return float("inf")
        return float(
            sum(
                float(amount) * base ** (-bond.frequency * float(time))
                for time, amount in zip(times, amounts, strict=True)
            )
        )

    dirty = scalar_price(yield_rate)
    base = 1.0 + yield_rate / bond.frequency
    present_values = [
        float(amount) * base ** (-bond.frequency * float(time))
        for time, amount in zip(times, amounts, strict=True)
    ]
    macaulay = (
        sum(float(time) * value for time, value in zip(times, present_values, strict=True)) / dirty
    )
    modified = macaulay / base
    convexity_numerator = sum(
        (bond.frequency * float(time)) * (bond.frequency * float(time) + 1.0) * value
        for time, value in zip(times, present_values, strict=True)
    )
    convexity = convexity_numerator / (dirty * bond.frequency**2 * base**2)
    up = scalar_price(yield_rate + bump_size)
    if yield_rate - bump_size > -bond.frequency:
        down = scalar_price(yield_rate - bump_size)
        dv01 = (down - up) / (2.0 * bump_size) * 1.0e-4
    else:
        dv01 = (dirty - up) / bump_size * 1.0e-4
    accrued = bond.accrued_interest(settlement)
    return ReferenceBondAnalytics(
        dirty_price=dirty,
        clean_price=dirty - accrued,
        accrued_interest=accrued,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
        dv01=dv01,
    )


@dataclass(frozen=True, slots=True)
class GoldenBondCase:
    """Immutable reference point for regression testing."""

    name: str
    maturity: float
    coupon_rate: float
    frequency: int
    yield_rate: float
    dirty_price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float


GOLDEN_BOND_CASES: tuple[GoldenBondCase, ...] = (
    GoldenBondCase(
        "five-year zero",
        5.0,
        0.0,
        2,
        0.04,
        82.03482998751551,
        5.0,
        4.901960784313726,
        26.432141484044603,
    ),
    GoldenBondCase(
        "ten-year par",
        10.0,
        0.05,
        2,
        0.05,
        100.00000000000014,
        7.989445671393991,
        7.794581142823406,
        73.6287314265636,
    ),
    GoldenBondCase(
        "final stub",
        1.25,
        0.12,
        2,
        0.08,
        104.69656952308132,
        1.1954255528860471,
        1.1494476470058146,
        1.9028147018119235,
    ),
)


def quantlib_available() -> bool:
    """Return whether the optional QuantLib independent engine is installed."""

    return find_spec("QuantLib") is not None


def _ql_date(ql: Any, value: DateLike) -> Any:
    current = as_date(value)
    return ql.Date(current.day, current.month, current.year)


def _build_ql_schedule(ql: Any, bond: FixedRateBond) -> Any:
    assert bond.issue_date is not None and bond.maturity_date is not None
    frequencies = {
        1: ql.Annual,
        2: ql.Semiannual,
        3: ql.EveryFourthMonth,
        4: ql.Quarterly,
        6: ql.Bimonthly,
        12: ql.Monthly,
    }
    if bond.frequency not in frequencies:
        raise ValueError("QuantLib validation requires a coupon frequency that divides 12")
    if bond.calendar.weekend_days != frozenset({5, 6}):
        raise ValueError("QuantLib validation supports the standard Saturday/Sunday weekend")
    calendar = ql.WeekendsOnly()
    for holiday in bond.calendar.holidays:
        calendar.addHoliday(_ql_date(ql, holiday))
    conventions = {
        BusinessDayConvention.UNADJUSTED: ql.Unadjusted,
        BusinessDayConvention.FOLLOWING: ql.Following,
        BusinessDayConvention.MODIFIED_FOLLOWING: ql.ModifiedFollowing,
        BusinessDayConvention.PRECEDING: ql.Preceding,
        BusinessDayConvention.MODIFIED_PRECEDING: ql.ModifiedPreceding,
    }
    rules = {
        "forward": ql.DateGeneration.Forward,
        "backward": ql.DateGeneration.Backward,
    }
    first = ql.Date() if bond.first_coupon_date is None else _ql_date(ql, bond.first_coupon_date)
    penultimate = (
        ql.Date()
        if bond.next_to_last_coupon_date is None
        else _ql_date(ql, bond.next_to_last_coupon_date)
    )
    return ql.Schedule(
        _ql_date(ql, bond.issue_date),
        _ql_date(ql, bond.maturity_date),
        ql.Period(frequencies[bond.frequency]),
        calendar,
        conventions[bond.business_day_convention],
        conventions[bond.termination_convention],
        rules[bond.date_generation],
        bond.end_of_month,
        first,
        penultimate,
    )


def quantlib_bond_schedule(bond: FixedRateBond) -> tuple[date, ...]:
    """Generate the bond schedule independently with optional QuantLib."""

    if not bond.is_dated:
        raise ValueError("QuantLib validation requires a dated bond")
    if not quantlib_available():
        raise ImportError("QuantLib is not installed; install qfin-quantum[validation]")
    ql = import_module("QuantLib")
    schedule = _build_ql_schedule(ql, bond)
    return tuple(
        date(int(item.year()), int(item.month()), int(item.dayOfMonth())) for item in schedule
    )


def quantlib_bond_from_yield(
    bond: FixedRateBond,
    yield_rate: float,
    *,
    settlement: DateLike,
) -> ReferenceBondAnalytics:
    """Return an optional independent QuantLib valuation for a dated bond.

    QuantLib is deliberately an optional validation dependency, not a runtime
    pricing dependency.  This adapter supports the conventions implemented by
    QFin 1.1 and uses QFin's generated date boundaries as explicit inputs.
    """

    if not bond.is_dated:
        raise ValueError("QuantLib validation requires a dated bond")
    if not quantlib_available():
        raise ImportError("QuantLib is not installed; install qfin-quantum[validation]")
    ql = import_module("QuantLib")
    day_counts = {
        DayCountConvention.ACT_365_FIXED: ql.Actual365Fixed(),
        DayCountConvention.ACT_360: ql.Actual360(),
        DayCountConvention.ACT_ACT: ql.ActualActual(ql.ActualActual.ISDA),
        DayCountConvention.THIRTY_360: ql.Thirty360(ql.Thirty360.USA),
        DayCountConvention.THIRTY_E_360: ql.Thirty360(ql.Thirty360.European),
    }
    conventions = {
        BusinessDayConvention.UNADJUSTED: ql.Unadjusted,
        BusinessDayConvention.FOLLOWING: ql.Following,
        BusinessDayConvention.MODIFIED_FOLLOWING: ql.ModifiedFollowing,
        BusinessDayConvention.PRECEDING: ql.Preceding,
        BusinessDayConvention.MODIFIED_PRECEDING: ql.ModifiedPreceding,
    }
    frequencies = {
        1: ql.Annual,
        2: ql.Semiannual,
        3: ql.EveryFourthMonth,
        4: ql.Quarterly,
        6: ql.Bimonthly,
        12: ql.Monthly,
    }
    schedule = _build_ql_schedule(ql, bond)
    day_counter = day_counts[bond.day_count]
    assert bond.issue_date is not None
    ql_bond = ql.FixedRateBond(
        0,
        bond.face_value,
        schedule,
        [bond.coupon_rate],
        day_counter,
        conventions[bond.business_day_convention],
        bond.face_value,
        _ql_date(ql, bond.issue_date),
    )
    settlement_date = _ql_date(ql, settlement)
    frequency = frequencies[bond.frequency]
    clean = float(
        ql.BondFunctions.cleanPrice(
            ql_bond,
            yield_rate,
            day_counter,
            ql.Compounded,
            frequency,
            settlement_date,
        )
    )
    accrued = float(ql_bond.accruedAmount(settlement_date))
    macaulay = float(
        ql.BondFunctions.duration(
            ql_bond,
            yield_rate,
            day_counter,
            ql.Compounded,
            frequency,
            ql.Duration.Macaulay,
            settlement_date,
        )
    )
    modified = float(
        ql.BondFunctions.duration(
            ql_bond,
            yield_rate,
            day_counter,
            ql.Compounded,
            frequency,
            ql.Duration.Modified,
            settlement_date,
        )
    )
    convexity = float(
        ql.BondFunctions.convexity(
            ql_bond,
            yield_rate,
            day_counter,
            ql.Compounded,
            frequency,
            settlement_date,
        )
    )
    up = float(
        ql.BondFunctions.cleanPrice(
            ql_bond,
            yield_rate + 1.0e-4,
            day_counter,
            ql.Compounded,
            frequency,
            settlement_date,
        )
    )
    down = float(
        ql.BondFunctions.cleanPrice(
            ql_bond,
            yield_rate - 1.0e-4,
            day_counter,
            ql.Compounded,
            frequency,
            settlement_date,
        )
    )
    return ReferenceBondAnalytics(
        dirty_price=clean + accrued,
        clean_price=clean,
        accrued_interest=accrued,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
        dv01=0.5 * (down - up),
    )


__all__ = [
    "GOLDEN_BOND_CASES",
    "FinancialTolerance",
    "FinancialValidationCheck",
    "FinancialValidationError",
    "FinancialValidationReport",
    "GoldenBondCase",
    "ReferenceBondAnalytics",
    "quantlib_available",
    "quantlib_bond_from_yield",
    "quantlib_bond_schedule",
    "reference_bond_from_yield",
    "validate_financial_values",
]
