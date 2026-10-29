"""Shared pytest fixtures."""

import pandas as pd
import pytest
import xarray as xr

from knmidata._auth import get_anonymous_opendata_api_key
from knmidata._http import AsyncTransport

S3 = "https://fake-s3.amazonaws.com"


@pytest.fixture(autouse=True)
def _clean_knmi_env(monkeypatch):
    """Ensure KNMI env vars never leak between tests and the key cache is fresh."""
    for var in ("KNMI_API_KEY", "KNMI_OPEN_DATA_KEY", "KNMI_EDR_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_anonymous_opendata_api_key.cache_clear()


@pytest.fixture
async def transport():
    """A transport with fast retry settings, closed after the test."""
    t = AsyncTransport(max_concurrency=4, timeout=5.0, max_retries=2, retry_backoff=0.0)
    yield t
    await t.aclose()


@pytest.fixture
def netcdf_bytes(tmp_path):
    """Bytes of a small real NetCDF file, as served from a presigned S3 URL."""
    path = tmp_path / "src" / "f1.nc"
    path.parent.mkdir()
    ds = xr.Dataset(
        {"temp": ("time", [1.0])},
        coords={"time": pd.to_datetime(["2024-01-01T00:00:00"])},
    )
    ds.to_netcdf(path, engine="h5netcdf")
    return path.read_bytes()


def coveragejson_payload() -> dict:
    """A minimal single-coverage CoverageJSON payload with t + station axes."""
    return {
        "type": "Coverage",
        "domain": {
            "axes": {
                "t": {"values": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]},
                "composite": {"values": ["260"]},
            },
            "referencing": [{"id": "EPSG:4326"}],
        },
        "parameters": {
            "T": {
                "unit": {"symbol": "C"},
                "description": {"en": "Temperature"},
            }
        },
        "ranges": {"T": {"values": [5.0, 6.0]}},
    }
