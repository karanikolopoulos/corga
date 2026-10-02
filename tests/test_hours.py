from datetime import date

from corga.data.hours import OpeningHours, access, parse_range
from corga.data.timeutil import at

TUE, WED = date(2026, 10, 6), date(2026, 10, 7)


def test_overnight_range_ends_next_day():
    assert parse_range(WED, "20:00-02:00").end == at(date(2026, 10, 8), "02:00")


def test_continuous_range_is_24_hours():
    interval = parse_range(WED, "00:00-24:00")
    assert (interval.end - interval.start).total_seconds() == 24 * 3600


def test_null_weekday_is_closed():
    hours = OpeningHours({"tue": None, "wed": "09:00-15:30"})
    assert access(hours, at(TUE, "11:00"), at(TUE, "12:00")) == "closed"
    assert not hours.is_open_on(TUE)
    assert hours.is_open_on(WED)


def test_null_hours_are_unknown_not_open():
    assert access(None, at(WED, "11:00"), at(WED, "12:00")) == "unknown"


def test_visit_must_finish_before_closing():
    hours = OpeningHours({"wed": "10:00-16:00"})
    assert access(hours, at(WED, "15:00"), at(WED, "16:00")) == "open"
    assert access(hours, at(WED, "15:30"), at(WED, "16:10")) == "closed"


def test_overnight_spill_from_previous_day_counts():
    hours = OpeningHours({"tue": "20:00-02:00", "wed": None})
    assert access(hours, at(WED, "00:30"), at(WED, "01:30")) == "open"


def test_earliest_start_waits_for_opening_or_gives_up():
    hours = OpeningHours({"wed": "10:00-16:00"})
    assert hours.earliest_start(at(WED, "09:40"), 60) == at(WED, "10:00")
    assert hours.earliest_start(at(WED, "15:30"), 60) is None
