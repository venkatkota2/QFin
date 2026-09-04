from datetime import date

import pytest

import qfin


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("ACT/365 Fixed", 366 / 365),
        ("ACT/360", 366 / 360),
        ("30/360", 1.0),
        ("30E/360", 1.0),
    ],
)
def test_day_count_conventions_across_leap_year(
    convention: str, expected: float
) -> None:
    result = qfin.year_fraction(date(2019, 12, 31), date(2020, 12, 31), convention)
    assert result == pytest.approx(expected)


def test_actual_actual_isda_splits_calendar_years() -> None:
    expected = 184 / 365 + 182 / 366
    assert qfin.year_fraction("2019-07-01", "2020-07-01", "ACT/ACT") == pytest.approx(
        expected
    )


def test_thirty_360_us_and_european_month_end_rules_differ() -> None:
    start = date(2021, 2, 28)
    end = date(2021, 3, 31)
    assert qfin.day_count(start, end, "30/360") == 30
    assert qfin.day_count(start, end, "30E/360") == 32


def test_day_counts_are_antisymmetric_and_aliases_are_explicit() -> None:
    forward = qfin.year_fraction("2024-02-29", "2025-02-28", "ACT/365F")
    backward = qfin.year_fraction("2025-02-28", "2024-02-29", "ACT/365 Fixed")
    assert backward == pytest.approx(-forward)
    assert qfin.DayCountConvention.parse("ACTUAL/360") is qfin.DayCountConvention.ACT_360
    with pytest.raises(ValueError, match="day-count"):
        qfin.year_fraction("2024-01-01", "2025-01-01", "BUS/252")
