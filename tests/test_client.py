"""End-to-end tests for the async and sync client facades."""

import httpx
import pytest
import respx

import knmidata
from knmidata._exceptions import NoFilesFoundError
from knmidata.config import EDR_BASE_URL as EDR
from knmidata.config import OPENDATA_BASE_URL as OPENDATA

S3 = "https://fake-s3.amazonaws.com"
FILES_URL = f"{OPENDATA}/datasets/ds/versions/1.0/files"


def _mock_open_data(router, filenames, netcdf_bytes):
    router.get(FILES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "isTruncated": False,
                "files": [{"filename": f} for f in filenames],
                "resultCount": len(filenames),
                "maxResults": 1000,
                "startAfterFilename": "",
                "nextPageToken": "",
            },
        )
    )
    for fname in filenames:
        router.get(f"{FILES_URL}/{fname}/url").mock(
            return_value=httpx.Response(
                200,
                json={
                    "contentType": "application/x-netcdf",
                    "size": str(len(netcdf_bytes)),
                    "lastModified": "2024-01-01T00:00:00Z",
                    "temporaryDownloadUrl": f"{S3}/{fname}",
                },
            )
        )
        router.get(f"{S3}/{fname}").mock(
            return_value=httpx.Response(200, content=netcdf_bytes)
        )


async def test_async_open_dataset_loads_into_memory(netcdf_bytes):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.open_dataset("ds", "1.0")
    assert "temp" in ds
    assert ds["temp"].values.tolist() == [1.0]


async def test_async_open_dataset_with_cache_dir(netcdf_bytes, tmp_path):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.open_dataset("ds", "1.0", cache_dir=tmp_path)
    assert (tmp_path / "f1.nc").exists()
    assert "temp" in ds


async def test_open_dataset_no_files_raises(netcdf_bytes):
    with respx.mock(assert_all_called=False) as router:
        router.get(FILES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "isTruncated": False,
                    "files": [],
                    "resultCount": 0,
                    "maxResults": 1000,
                    "startAfterFilename": "",
                    "nextPageToken": "",
                },
            )
        )
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            with pytest.raises(NoFilesFoundError):
                await client.open_dataset("ds", "1.0")


async def test_open_dataset_max_files_and_pattern(netcdf_bytes, tmp_path):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["a.nc", "b.nc", "c.txt"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.open_dataset(
                "ds",
                "1.0",
                cache_dir=tmp_path / "cache",
                file_pattern=r"\.nc$",
                max_files=1,
            )
        s3_calls = [c for c in router.calls if "fake-s3" in str(c.request.url)]
    assert len(s3_calls) == 1
    assert str(s3_calls[0].request.url).endswith("/a.nc")
    assert (tmp_path / "cache" / "a.nc").exists()
    assert not (tmp_path / "cache" / "c.txt").exists()
    assert "temp" in ds


def test_sync_client_can_be_called_repeatedly(netcdf_bytes):
    """Regression test: KNMIClient used to fail on the second call because
    asyncio.run() created a new event loop per call while the transport
    semaphore stayed bound to the first loop."""
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        with knmidata.KNMIClient(api_key="key") as client:
            ds1 = client.open_dataset("ds", "1.0")
            ds2 = client.open_dataset("ds", "1.0")
    assert ds1["temp"].values.tolist() == [1.0]
    assert ds2["temp"].values.tolist() == [1.0]


def test_sync_edr_facade(netcdf_bytes):
    from conftest import coveragejson_payload

    endpoint = f"{EDR}/collections/c1/position"
    with respx.mock(assert_all_called=False) as router:
        router.get(endpoint).mock(
            return_value=httpx.Response(200, json=coveragejson_payload())
        )
        with knmidata.KNMIClient(api_key="key", edr_key="edr-key") as client:
            ds = client.edr.position("c1", coords=(4.89, 52.37))
    assert ds["T"].values.tolist() == [[5.0], [6.0]]


def test_module_level_open_dataset(netcdf_bytes):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        ds = knmidata.open_dataset("ds", "1.0", api_key="key")
    assert ds["temp"].values.tolist() == [1.0]


def test_version_and_exports():
    assert isinstance(knmidata.__version__, str)
    for name in knmidata.__all__:
        assert hasattr(knmidata, name)
