"""Tests for the Open Data catalog client."""

from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest
import respx

from knmidata._auth import APIKeyAuth
from knmidata._exceptions import KNMIAuthError, KNMIError
from knmidata.config import OPENDATA_BASE_URL as OPENDATA
from knmidata.opendata.client import AsyncOpenDataClient

FILES_URL = f"{OPENDATA}/datasets/ds/versions/1.0/files"


@pytest.fixture
def catalog(transport):
    return AsyncOpenDataClient(transport=transport, auth=APIKeyAuth(api_key="key"))


def _page(filenames, token="", truncated=False):
    return {
        "isTruncated": truncated,
        "files": [{"filename": f} for f in filenames],
        "resultCount": len(filenames),
        "maxResults": 1000,
        "startAfterFilename": "",
        "nextPageToken": token,
    }


async def test_list_files_paginates_and_dedups(catalog):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(FILES_URL).mock(
            side_effect=[
                httpx.Response(
                    200, json=_page(["a.nc", "b.nc"], token="tok", truncated=True)
                ),
                httpx.Response(200, json=_page(["b.nc", "c.nc"])),
            ]
        )
        files = await catalog.list_files("ds", "1.0")
        assert files == ["a.nc", "b.nc", "c.nc"]
        assert route.call_count == 2
        assert route.calls[1].request.url.params["nextPageToken"] == "tok"


async def test_list_files_detects_pagination_loop(catalog):
    with respx.mock(assert_all_called=False) as router:
        router.get(FILES_URL).mock(
            return_value=httpx.Response(
                200, json=_page(["a.nc"], token="tok", truncated=True)
            )
        )
        with pytest.raises(KNMIError, match="Pagination loop"):
            await catalog.list_files("ds", "1.0")


async def test_list_files_sends_auth_header(catalog):
    def check_auth(request):
        assert request.headers["Authorization"] == "key"
        return httpx.Response(200, json=_page([]))

    with respx.mock(assert_all_called=False) as router:
        router.get(FILES_URL).mock(side_effect=check_auth)
        await catalog.list_files("ds", "1.0")


async def test_time_filter_uses_last_modified_ordering(catalog):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(FILES_URL).mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        await catalog.list_files(
            "ds",
            "1.0",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-02-01T00:00:00Z",
        )
        params = route.calls[0].request.url.params
        assert params["begin"] == "2024-01-01T00:00:00Z"
        assert params["end"] == "2024-02-01T00:00:00Z"
        assert params["orderBy"] == "lastModified"
        assert params["maxKeys"] == "1000"


async def test_no_time_filter_keeps_filename_ordering(catalog):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(FILES_URL).mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        await catalog.list_files("ds", "1.0")
        assert "orderBy" not in route.calls[0].request.url.params
        assert "begin" not in route.calls[0].request.url.params


async def test_file_pattern_filters_client_side(catalog):
    with respx.mock(assert_all_called=False) as router:
        router.get(FILES_URL).mock(
            return_value=httpx.Response(200, json=_page(["a.nc", "b.txt", "c.nc"]))
        )
        files = await catalog.list_files("ds", "1.0", file_pattern=r"\.nc$")
        assert files == ["a.nc", "c.nc"]


async def test_explicit_order_by_is_respected(catalog):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(FILES_URL).mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        await catalog.list_files(
            "ds", "1.0", start_time="2024-01-01T00:00:00Z", order_by="created"
        )
        assert route.calls[0].request.url.params["orderBy"] == "created"


async def test_timestamp_inputs_normalized_to_utc(catalog):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(FILES_URL).mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        await catalog.list_files(
            "ds",
            "1.0",
            start_time=pd.Timestamp("2024-01-01 12:00"),  # naive -> UTC
            end_time=pd.Timestamp("2024-01-02T12:00:00+02:00"),  # -> UTC
        )
        params = route.calls[0].request.url.params
        assert params["begin"] == "2024-01-01T12:00:00Z"
        assert params["end"] == "2024-01-02T10:00:00Z"
        assert params["orderBy"] == "lastModified"


async def test_datetime_objects_accepted(catalog):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(FILES_URL).mock(
            return_value=httpx.Response(200, json=_page([]))
        )
        await catalog.list_files(
            "ds", "1.0", start_time=datetime(2024, 1, 1, tzinfo=UTC)
        )
        assert route.calls[0].request.url.params["begin"] == "2024-01-01T00:00:00Z"


async def test_start_after_end_raises(catalog):
    with (
        respx.mock(assert_all_called=False),
        pytest.raises(ValueError, match="after end_time"),
    ):
        await catalog.list_files(
            "ds",
            "1.0",
            start_time="2024-02-01T00:00:00Z",
            end_time="2024-01-01T00:00:00Z",
        )


async def test_403_raises_auth_error(catalog):
    with respx.mock(assert_all_called=False) as router:
        router.get(FILES_URL).mock(return_value=httpx.Response(403, text="forbidden"))
        with pytest.raises(KNMIAuthError):
            await catalog.list_files("ds", "1.0")
