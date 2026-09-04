"""Explicit financial day-count conventions."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from qfin.finance.dates import DateLike, as_date, is_month_end


class DayCountConvention(StrEnum):
    """Day-count conventions implemented by QFin."""

    ACT_365_FIXED = "ACT/365 Fixed"
    ACT_360 = "ACT/360"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"
    THIRTY_E_360 = "30E/360"

    @classmethod
    def parse(cls, value: DayCountConvention | str) -> DayCountConvention:
        if isinstance(value, cls):
            return value
        normalized = "".join(str(value).upper().split()).replace("_", "/")
        aliases = {
            "ACT/365": cls.ACT_365_FIXED,
            "ACT/365F": cls.ACT_365_FIXED,
            "ACTUAL/365": cls.ACT_365_FIXED,
            "ACTUAL/365FIXED": cls.ACT_365_FIXED,
            "ACTUAL/360": cls.ACT_360,
            "ACTUAL/ACTUAL": cls.ACT_ACT,
            "ACT/ACTISDA": cls.ACT_ACT,
            "30U/360": cls.THIRTY_360,
            "30US/360": cls.THIRTY_360,
            "30/360US": cls.THIRTY_360,
            "30E/360": cls.THIRTY_E_360,
        }
        for item in cls:
            aliases["".join(item.value.upper().split())] = item
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"day-count convention must be one of: {choices}") from exc


def _days_in_year(year: int) -> int:
    return (date(year + 1, 1, 1) - date(year, 1, 1)).days


def _actual_actual_isda(start: date, end: date) -> float:
    if start.year == end.year:
        return (end - start).days / _days_in_year(start.year)
    result = (date(start.year + 1, 1, 1) - start).days / _days_in_year(start.year)
    for _year in range(start.year + 1, end.year):
        result += 1.0
    result += (end - date(end.year, 1, 1)).days / _days_in_year(end.year)
    return result


def _thirty_360_us(start: date, end: date) -> int:
    start_february_end = start.month == 2 and is_month_end(start)
    end_february_end = end.month == 2 and is_month_end(end)
    start_day = 30 if start.day == 31 or start_february_end else start.day
    end_day = end.day
    if (start_february_end and end_february_end) or (end_day == 31 and start_day == 30):
        end_day = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + end_day - start_day


def _thirty_e_360(start: date, end: date) -> int:
    return (
        360 * (end.year - start.year)
        + 30 * (end.month - start.month)
        + min(end.day, 30)
        - min(start.day, 30)
    )


def day_count(
    start_date: DateLike,
    end_date: DateLike,
    convention: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
) -> int:
    """Return the signed day-count numerator for the selected convention."""

    start = as_date(start_date, name="start date")
    end = as_date(end_date, name="end date")
    if start == end:
        return 0
    sign = 1
    if end < start:
        start, end = end, start
        sign = -1
    selected = DayCountConvention.parse(convention)
    if selected in (
        DayCountConvention.ACT_365_FIXED,
        DayCountConvention.ACT_360,
        DayCountConvention.ACT_ACT,
    ):
        result = (end - start).days
    elif selected is DayCountConvention.THIRTY_360:
        result = _thirty_360_us(start, end)
    else:
        result = _thirty_e_360(start, end)
    return sign * result


def year_fraction(
    start_date: DateLike,
    end_date: DateLike,
    convention: DayCountConvention | str = DayCountConvention.ACT_365_FIXED,
) -> float:
    """Return the signed year fraction under a named financial convention.

    ``ACT/ACT`` is the ISDA year-splitting convention and ``30/360`` is the
    US/NASD convention.  Those definitions are explicit to avoid hidden market
    assumptions.
    """

    start = as_date(start_date, name="start date")
    end = as_date(end_date, name="end date")
    if start == end:
        return 0.0
    sign = 1.0
    if end < start:
        start, end = end, start
        sign = -1.0
    selected = DayCountConvention.parse(convention)
    if selected is DayCountConvention.ACT_365_FIXED:
        result = (end - start).days / 365.0
    elif selected is DayCountConvention.ACT_360:
        result = (end - start).days / 360.0
    elif selected is DayCountConvention.ACT_ACT:
        result = _actual_actual_isda(start, end)
    elif selected is DayCountConvention.THIRTY_360:
        result = _thirty_360_us(start, end) / 360.0
    else:
        result = _thirty_e_360(start, end) / 360.0
    return sign * result


__all__ = ["DayCountConvention", "day_count", "year_fraction"]
