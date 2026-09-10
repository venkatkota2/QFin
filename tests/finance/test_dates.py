from datetime import date

import pytest

import qfin


def test_calendar_business_day_adjustments_and_holidays() -> None:
    calendar = qfin.Calendar(
        "Toronto test calendar",
        holidays=frozenset({date(2026, 7, 1)}),
    )
    assert not calendar.is_business_day("2026-07-01")
    assert calendar.adjust("2026-07-01", "following") == date(2026, 7, 2)
    assert calendar.adjust("2026-05-31", "modified_following") == date(2026, 5, 29)
    assert calendar.adjust("2026-08-01", "preceding") == date(2026, 7, 31)
    assert calendar.advance_business_days("2026-06-30", 2) == date(2026, 7, 3)


def test_month_end_and_leap_year_schedule() -> None:
    schedule = qfin.Schedule(
        "2024-01-31",
        "2025-01-31",
        4,
        business_day_convention="unadjusted",
        end_of_month=True,
        date_generation="forward",
    )
    assert schedule.unadjusted_dates == (
        date(2024, 1, 31),
        date(2024, 4, 30),
        date(2024, 7, 31),
        date(2024, 10, 31),
        date(2025, 1, 31),
    )
    assert qfin.add_months("2024-02-29", 12, end_of_month=True) == date(2025, 2, 28)
    assert qfin.is_month_end("2024-02-29")


def test_schedule_supports_explicit_short_and_long_stub_boundaries() -> None:
    schedule = qfin.Schedule(
        "2026-01-15",
        "2028-01-31",
        2,
        business_day_convention="unadjusted",
        first_coupon_date="2026-10-31",
        next_to_last_coupon_date="2027-10-31",
    )
    assert schedule.unadjusted_dates == (
        date(2026, 1, 15),
        date(2026, 10, 31),
        date(2027, 4, 30),
        date(2027, 10, 31),
        date(2028, 1, 31),
    )


def test_schedule_and_valuation_date_reject_ambiguous_inputs() -> None:
    assert str(qfin.ValuationDate("2026-09-04")) == "2026-09-04"
    with pytest.raises(ValueError, match="after"):
        qfin.Schedule("2026-01-01", "2026-01-01", 2)
    with pytest.raises(ValueError, match="frequency"):
        qfin.Schedule("2026-01-01", "2027-01-01", 5)
    with pytest.raises(ValueError, match="ISO"):
        qfin.ValuationDate("09/04/2026")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"date_generation": "nearest"}, "date_generation"),
        ({"first_coupon_date": "2026-01-01"}, "first coupon date"),
        ({"first_coupon_date": "2028-01-01"}, "first coupon date"),
        ({"next_to_last_coupon_date": "2026-01-01"}, "next-to-last coupon date"),
        ({"next_to_last_coupon_date": "2028-01-01"}, "next-to-last coupon date"),
        ({"first_coupon_date": "2027-10-01", "next_to_last_coupon_date": "2027-01-01"},
         "must not follow"),
    ],
)
def test_schedule_rejects_inconsistent_stub_contracts(
    kwargs: dict[str, str], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        qfin.Schedule("2026-01-01", "2028-01-01", 2, **kwargs)


def test_forward_schedule_keeps_both_explicit_stubs() -> None:
    schedule = qfin.Schedule(
        "2026-01-15", "2028-01-31", 2, date_generation="forward",
        business_day_convention="unadjusted", first_coupon_date="2026-10-31",
        next_to_last_coupon_date="2027-10-15",
    )
    expected = (date(2026, 1, 15), date(2026, 10, 31), date(2027, 4, 30),
                date(2027, 10, 15), date(2028, 1, 31))
    assert tuple(schedule) == expected
    assert len(schedule) == 5
    assert schedule.payment_dates == expected[1:]


def test_calendar_adjustments_cannot_collapse_coupon_periods() -> None:
    # Saturday and Sunday both adjust to Monday: a zero-length coupon period
    # must be rejected rather than flow into accrued interest or a yield solve.
    with pytest.raises(ValueError, match="non-increasing"):
        qfin.Schedule("2026-05-30", "2026-05-31", 2,
                      business_day_convention="following")
    with pytest.raises(ValueError, match="name"):
        qfin.Calendar(name=" ")
