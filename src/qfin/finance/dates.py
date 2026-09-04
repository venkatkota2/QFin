"""Financial dates, business-day calendars, and coupon schedules."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from operator import index as integer_index
from typing import TypeAlias

DateLike: TypeAlias = date | datetime | str


def as_date(value: DateLike, *, name: str = "date") -> date:
    """Normalize a standard-library date, datetime, or ISO date string."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid ISO date") from exc
    raise TypeError(f"{name} must be a date, datetime, or ISO date string")


def is_month_end(value: DateLike) -> bool:
    """Return whether ``value`` is the final calendar day of its month."""

    current = as_date(value)
    return (current + timedelta(days=1)).month != current.month


def add_months(value: DateLike, months: int, *, end_of_month: bool = False) -> date:
    """Add calendar months with deterministic month-end handling."""

    current = as_date(value)
    try:
        month_count = integer_index(months)
    except TypeError as exc:
        raise ValueError("months must be an integer") from exc
    if isinstance(months, bool):
        raise ValueError("months must be an integer")
    absolute_month = current.year * 12 + current.month - 1 + month_count
    year, zero_based_month = divmod(absolute_month, 12)
    month = zero_based_month + 1
    if not 1 <= year <= 9999:
        raise ValueError("resulting date is outside the supported year range")
    if year == 9999 and month == 12:
        last_day = 31
    else:
        first_next_month = date(year + (month == 12), month % 12 + 1, 1)
        last_day = (first_next_month - timedelta(days=1)).day
    preserve_month_end = end_of_month and is_month_end(current)
    day = last_day if preserve_month_end else min(current.day, last_day)
    return date(year, month, day)


class BusinessDayConvention(StrEnum):
    """Supported business-day adjustment conventions."""

    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"

    @classmethod
    def parse(cls, value: BusinessDayConvention | str) -> BusinessDayConvention:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "none": cls.UNADJUSTED,
            "modifiedfollowing": cls.MODIFIED_FOLLOWING,
            "modifiedpreceding": cls.MODIFIED_PRECEDING,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"business-day convention must be one of: {choices}") from exc


@dataclass(frozen=True, slots=True)
class Calendar:
    """Weekend and explicit-holiday business calendar.

    QFin intentionally uses :class:`datetime.date` for the underlying date
    representation.  Holiday sets are supplied by the caller, so no regulatory
    or market calendar is silently assumed.
    """

    name: str = "weekends-only"
    holidays: frozenset[date] = field(default_factory=frozenset)
    weekend_days: frozenset[int] = field(default_factory=lambda: frozenset({5, 6}))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("calendar name must not be empty")
        holidays = frozenset(as_date(item, name="holiday") for item in self.holidays)
        try:
            weekend_days = frozenset(integer_index(item) for item in self.weekend_days)
        except TypeError as exc:
            raise ValueError("weekend days must be integers from 0 through 6") from exc
        if any(item < 0 or item > 6 for item in weekend_days):
            raise ValueError("weekend days must be integers from 0 through 6")
        object.__setattr__(self, "holidays", holidays)
        object.__setattr__(self, "weekend_days", weekend_days)

    def is_business_day(self, value: DateLike) -> bool:
        current = as_date(value)
        return current.weekday() not in self.weekend_days and current not in self.holidays

    def adjust(
        self,
        value: DateLike,
        convention: BusinessDayConvention | str = BusinessDayConvention.FOLLOWING,
    ) -> date:
        """Adjust a date according to an explicit business-day convention."""

        current = as_date(value)
        selected = BusinessDayConvention.parse(convention)
        if selected is BusinessDayConvention.UNADJUSTED or self.is_business_day(current):
            return current

        def seek(direction: int) -> date:
            candidate = current
            while not self.is_business_day(candidate):
                candidate += timedelta(days=direction)
            return candidate

        if selected in (
            BusinessDayConvention.FOLLOWING,
            BusinessDayConvention.MODIFIED_FOLLOWING,
        ):
            adjusted = seek(1)
            if (
                selected is BusinessDayConvention.MODIFIED_FOLLOWING
                and adjusted.month != current.month
            ):
                return seek(-1)
            return adjusted
        adjusted = seek(-1)
        if (
            selected is BusinessDayConvention.MODIFIED_PRECEDING
            and adjusted.month != current.month
        ):
            return seek(1)
        return adjusted

    def advance_business_days(self, value: DateLike, days: int) -> date:
        """Advance by a signed number of business days."""

        current = as_date(value)
        try:
            remaining = abs(integer_index(days))
        except TypeError as exc:
            raise ValueError("days must be an integer") from exc
        if isinstance(days, bool):
            raise ValueError("days must be an integer")
        direction = 1 if days >= 0 else -1
        while remaining:
            current += timedelta(days=direction)
            if self.is_business_day(current):
                remaining -= 1
        return current


@dataclass(frozen=True, slots=True)
class ValuationDate:
    """Typed valuation-date wrapper backed by :class:`datetime.date`."""

    value: date

    def __init__(self, value: DateLike) -> None:
        object.__setattr__(self, "value", as_date(value, name="valuation date"))

    def __str__(self) -> str:
        return self.value.isoformat()


@dataclass(frozen=True, slots=True, init=False)
class Schedule:
    """Regular dated schedule with explicit stub and adjustment rules."""

    start_date: date
    end_date: date
    frequency: int
    calendar: Calendar
    business_day_convention: BusinessDayConvention
    termination_convention: BusinessDayConvention
    date_generation: str
    end_of_month: bool
    first_coupon_date: date | None
    next_to_last_coupon_date: date | None
    unadjusted_dates: tuple[date, ...]
    dates: tuple[date, ...]

    def __init__(
        self,
        start_date: DateLike,
        end_date: DateLike,
        frequency: int,
        *,
        calendar: Calendar | None = None,
        business_day_convention: BusinessDayConvention | str = (
            BusinessDayConvention.MODIFIED_FOLLOWING
        ),
        termination_convention: BusinessDayConvention | str | None = None,
        date_generation: str = "backward",
        end_of_month: bool = False,
        first_coupon_date: DateLike | None = None,
        next_to_last_coupon_date: DateLike | None = None,
    ) -> None:
        start = as_date(start_date, name="start date")
        end = as_date(end_date, name="end date")
        if end <= start:
            raise ValueError("end date must be after start date")
        try:
            normalized_frequency = integer_index(frequency)
        except TypeError as exc:
            raise ValueError("frequency must divide 12") from exc
        if (
            isinstance(frequency, bool)
            or normalized_frequency <= 0
            or 12 % normalized_frequency != 0
        ):
            raise ValueError("frequency must be one of 1, 2, 3, 4, 6, or 12")
        generation = date_generation.strip().lower()
        if generation not in ("forward", "backward"):
            raise ValueError("date_generation must be 'forward' or 'backward'")
        first = (
            None
            if first_coupon_date is None
            else as_date(first_coupon_date, name="first coupon date")
        )
        penultimate = (
            None
            if next_to_last_coupon_date is None
            else as_date(next_to_last_coupon_date, name="next-to-last coupon date")
        )
        if first is not None and not start < first < end:
            raise ValueError("first coupon date must be between start and end dates")
        if penultimate is not None and not start < penultimate < end:
            raise ValueError("next-to-last coupon date must be between start and end dates")
        if first is not None and penultimate is not None and first > penultimate:
            raise ValueError("first coupon date must not follow next-to-last coupon date")

        selected_calendar = calendar or Calendar()
        convention = BusinessDayConvention.parse(business_day_convention)
        termination = BusinessDayConvention.parse(termination_convention or convention)
        months = 12 // normalized_frequency
        preserve_eom = end_of_month and (
            is_month_end(start) if generation == "forward" else is_month_end(end)
        )
        if generation == "forward":
            boundaries = [start]
            anchor = first or start
            if first is not None:
                boundaries.append(first)
            limit = penultimate or end
            cursor = add_months(anchor, months, end_of_month=preserve_eom)
            while cursor < limit:
                boundaries.append(cursor)
                cursor = add_months(cursor, months, end_of_month=preserve_eom)
            if penultimate is not None and boundaries[-1] != penultimate:
                boundaries.append(penultimate)
            boundaries.append(end)
        else:
            reverse_boundaries = [end]
            anchor = penultimate or end
            if penultimate is not None:
                reverse_boundaries.append(penultimate)
            limit = first or start
            cursor = add_months(anchor, -months, end_of_month=preserve_eom)
            while cursor > limit:
                reverse_boundaries.append(cursor)
                cursor = add_months(cursor, -months, end_of_month=preserve_eom)
            if first is not None and reverse_boundaries[-1] != first:
                reverse_boundaries.append(first)
            reverse_boundaries.append(start)
            boundaries = list(reversed(reverse_boundaries))

        unadjusted = tuple(dict.fromkeys(boundaries))
        adjusted = tuple(
            selected_calendar.adjust(
                item,
                termination if position == len(unadjusted) - 1 else convention,
            )
            for position, item in enumerate(unadjusted)
        )
        if any(left >= right for left, right in pairwise(adjusted)):
            raise ValueError("business-day adjustment produced non-increasing schedule dates")

        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "frequency", normalized_frequency)
        object.__setattr__(self, "calendar", selected_calendar)
        object.__setattr__(self, "business_day_convention", convention)
        object.__setattr__(self, "termination_convention", termination)
        object.__setattr__(self, "date_generation", generation)
        object.__setattr__(self, "end_of_month", end_of_month)
        object.__setattr__(self, "first_coupon_date", first)
        object.__setattr__(self, "next_to_last_coupon_date", penultimate)
        object.__setattr__(self, "unadjusted_dates", unadjusted)
        object.__setattr__(self, "dates", adjusted)

    @property
    def payment_dates(self) -> tuple[date, ...]:
        """Adjusted dates after the schedule start date."""

        return self.dates[1:]

    def __len__(self) -> int:
        return len(self.dates)

    def __iter__(self) -> Iterator[date]:
        return iter(self.dates)


__all__ = [
    "BusinessDayConvention",
    "Calendar",
    "DateLike",
    "Schedule",
    "ValuationDate",
    "add_months",
    "as_date",
    "is_month_end",
]
