"""Tests for TimeLike normalization to ISO-8601 UTC strings."""

from datetime import UTC, date, datetime, timedelta, timezone

import pandas as pd
import pytest

from knmidata._time import to_iso8601


def test_string_passthrough_normalized():
    assert to_iso8601("2024-01-01T00:00:00Z") == "2024-01-01T00:00:00Z"


def test_date_only_string_gets_time_and_zone():
    assert to_iso8601("2024-01-01") == "2024-01-01T00:00:00Z"


def test_naive_timestamp_assumed_utc():
    assert to_iso8601(pd.Timestamp("2024-01-01 12:30")) == "2024-01-01T12:30:00Z"


def test_tzaware_timestamp_converted_to_utc():
    ts = pd.Timestamp("2024-01-01T12:00:00+02:00")
    assert to_iso8601(ts) == "2024-01-01T10:00:00Z"


def test_datetime_and_date_objects():
    assert to_iso8601(datetime(2024, 1, 1, 6, 0, tzinfo=UTC)) == "2024-01-01T06:00:00Z"
    assert to_iso8601(
        datetime(2024, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=2)))
    ) == ("2024-01-01T06:00:00Z")
    assert to_iso8601(date(2024, 1, 1)) == "2024-01-01T00:00:00Z"


def test_fractional_seconds_kept():
    assert (
        to_iso8601(pd.Timestamp("2024-01-01T00:00:00.5"))
        == "2024-01-01T00:00:00.500000Z"
    )


def test_invalid_string_raises():
    with pytest.raises(ValueError):
        to_iso8601("not a timestamp")
