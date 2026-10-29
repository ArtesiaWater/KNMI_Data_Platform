"""Normalization of user-supplied time arguments to ISO-8601 UTC strings."""

from datetime import date, datetime
from typing import TypeAlias

import pandas as pd

TimeLike: TypeAlias = str | pd.Timestamp | datetime | date
"""Accepted time argument: ISO-8601 string, pandas Timestamp, datetime, or date."""


def to_iso8601(value: TimeLike) -> str:
    """Normalize a time-like value to an ISO-8601 string in UTC with a 'Z' suffix.

    Naive values are assumed to be UTC; timezone-aware values are converted to UTC.

    Parameters
    ----------
    value : TimeLike
        ISO-8601 string, pandas Timestamp, datetime, or date.

    Returns
    -------
    str
        Timestamp in UTC, e.g. '2024-01-01T00:00:00Z'.
    """
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")
