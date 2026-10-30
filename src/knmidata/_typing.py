"""Shared type aliases used across knmidata.

All custom type hints live here so they have a single definition site.
This module must not import other knmidata modules (no circular imports).
"""

from collections.abc import Callable, Coroutine, Generator, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

import httpx
import pandas as pd

T = TypeVar("T")

# --- time arguments ---------------------------------------------------------
TimeLike: TypeAlias = str | pd.Timestamp | datetime | date
"""Accepted time argument: ISO-8601 string, pandas Timestamp, datetime, or date."""

# --- paths ------------------------------------------------------------------
PathLike: TypeAlias = str | Path
"""Anything path-like: a string or pathlib.Path."""

FilePaths: TypeAlias = Sequence[PathLike]
"""A sequence of file paths (strings and/or pathlib.Path, mixed allowed)."""

# --- EDR query shapes -------------------------------------------------------
Coords: TypeAlias = tuple[float, float]
"""Point coordinates as (lon, lat)."""

BBox: TypeAlias = tuple[float, float, float, float]
"""Bounding box as (minx, miny, maxx, maxy)."""

ZLevel: TypeAlias = str | tuple[float, float]
"""Vertical level: single value ('100') or (min, max) interval."""

OutputFormat: TypeAlias = Literal["CoverageJSON", "GeoJSON"]
"""EDR data query output formats supported by the KNMI EDR API."""

# --- Open Data shapes -------------------------------------------------------
CombineMethod: TypeAlias = Literal["by_coords", "nested", "no_conflicts"]
"""How files_to_dataset combines per-file datasets."""

# --- HTTP plumbing ----------------------------------------------------------
QueryParams: TypeAlias = dict[str, Any]
"""Query parameters sent with a request."""

AuthFlow: TypeAlias = Generator[httpx.Request, httpx.Response, None]
"""Signature of an httpx.Auth.auth_flow generator."""

RunCoroutine: TypeAlias = Callable[[Coroutine[Any, Any, T]], T]
"""Scheduler running a coroutine to completion and returning its result."""
