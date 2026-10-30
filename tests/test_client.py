"""End-to-end tests for the async and sync client facades."""

import httpx
import pytest
import respx

import knmidata
from knmidata._exceptions import KNMIError, NoFilesFoundError
from knmidata.config import ANONYMOUS_KEY_DOCS_URL
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


async def test_async_get_dataset_loads_into_memory(netcdf_bytes):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.get_dataset("ds", "1.0")
    assert "temp" in ds
    assert ds["temp"].values.tolist() == [1.0]


async def test_async_get_dataset_with_cache_dir(netcdf_bytes, tmp_path):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.get_dataset("ds", "1.0", cache_dir=tmp_path)
    assert (tmp_path / "f1.nc").exists()
    assert "temp" in ds


async def test_get_dataset_no_files_raises(netcdf_bytes):
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
                await client.get_dataset("ds", "1.0")


async def test_get_dataset_max_files_and_pattern(netcdf_bytes, tmp_path):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["a.nc", "b.nc", "c.txt"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.get_dataset(
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
            ds1 = client.get_dataset("ds", "1.0")
            ds2 = client.get_dataset("ds", "1.0")
    assert ds1["temp"].values.tolist() == [1.0]
    assert ds2["temp"].values.tolist() == [1.0]


def test_sync_catalog_and_downloader_facades(netcdf_bytes, tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get(FILES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "isTruncated": False,
                    "files": [{"filename": "f1.nc"}],
                    "resultCount": 1,
                    "maxResults": 1000,
                    "startAfterFilename": "",
                    "nextPageToken": "",
                },
            )
        )
        url_endpoint = f"{FILES_URL}/f1.nc/url"
        router.get(url_endpoint).mock(
            return_value=httpx.Response(
                200,
                json={
                    "contentType": "application/x-netcdf",
                    "size": str(len(netcdf_bytes)),
                    "lastModified": "2024-01-01T00:00:00Z",
                    "temporaryDownloadUrl": f"{S3}/f1.nc",
                },
            )
        )
        router.get(f"{S3}/f1.nc").mock(
            return_value=httpx.Response(200, content=netcdf_bytes)
        )
        with knmidata.KNMIClient(api_key="key") as client:
            files = client.opendata.list_files("ds", "1.0")
            assert files == ["f1.nc"]
            assert (
                client.opendata.get_download_url("ds", "1.0", "f1.nc") == f"{S3}/f1.nc"
            )
            paths = client.opendata.download_files(
                "ds", "1.0", files, destination_dir=tmp_path / "out"
            )
            assert (tmp_path / "out" / "f1.nc").exists()
            assert paths[0].name == "f1.nc"


def test_sync_facade_completion_and_docstrings():
    """The sync facade must expose real attributes (tab-completion) and the
    original docstrings/signatures (help())."""
    import inspect

    import knmidata.edr.client as edr_mod
    from knmidata.client import _SyncFacade

    with knmidata.KNMIClient(api_key="key", edr_key="edr-key") as client:
        facade = client.edr
        assert isinstance(facade, _SyncFacade)
        # runtime tab-completion: wrapped methods show up in dir()
        assert "get_position" in dir(facade)
        assert "get_location" in dir(facade)
        # docstring and signature are the async method's own
        assert (
            facade.get_position.__doc__ == edr_mod.AsyncEDRClient.get_position.__doc__
        )
        facade_sig = inspect.signature(facade.get_position)
        async_sig = inspect.signature(edr_mod.AsyncEDRClient.get_position)
        assert list(facade_sig.parameters) == list(async_sig.parameters)[1:]
        assert facade_sig.return_annotation == async_sig.return_annotation
        assert facade.get_position.__wrapped__ is not None
        # the opendata namespace methods are bound on the facade too
        assert "list_files" in dir(client.opendata)
        assert "get_dataset" in dir(client.opendata)
        assert "open_dataset" in dir(client.opendata)
        assert client.opendata.list_files.__doc__


def test_sync_edr_facade(netcdf_bytes):
    from conftest import coveragejson_payload

    endpoint = f"{EDR}/collections/c1/position"
    with respx.mock(assert_all_called=False) as router:
        router.get(endpoint).mock(
            return_value=httpx.Response(200, json=coveragejson_payload())
        )
        with knmidata.KNMIClient(api_key="key", edr_key="edr-key") as client:
            ds = client.edr.get_position("c1", coords=(4.89, 52.37))
    assert ds["T"].values.tolist() == [[5.0], [6.0]]


def test_module_level_get_dataset(netcdf_bytes):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        ds = knmidata.get_dataset("ds", "1.0", api_key="key")
    assert ds["temp"].values.tolist() == [1.0]


async def test_get_dataset_forwards_combine_kwargs(netcdf_bytes, tmp_path):
    # both files carry the same time coordinate; combine="nested" concatenates
    # along concat_dim, which only works if the kwargs reach files_to_dataset
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc", "f2.nc"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            ds = await client.get_dataset(
                "ds", "1.0", cache_dir=tmp_path, combine="nested", concat_dim="time"
            )
    assert ds.sizes["time"] == 2


async def test_get_dataset_reuses_files_already_on_disk(netcdf_bytes, tmp_path):
    with respx.mock(assert_all_called=False) as router:
        _mock_open_data(router, ["f1.nc"], netcdf_bytes)
        async with knmidata.AsyncKNMIClient(api_key="key") as client:
            await client.get_dataset("ds", "1.0", cache_dir=tmp_path)
            await client.get_dataset("ds", "1.0", cache_dir=tmp_path)
        # read router.calls before the mock context exits (calls are cleared on stop)
        s3_calls = [c for c in router.calls if "fake-s3" in str(c.request.url)]
    assert len(s3_calls) == 1  # second call reused the cached file, no re-download


def test_sync_client_use_after_close_raises():
    with knmidata.KNMIClient(api_key="key") as client:
        pass
    with pytest.raises(KNMIError, match="closed"):
        client.get_dataset("ds", "1.0")
    client.close()  # idempotent


def test_sync_facades_are_cached():
    with knmidata.KNMIClient(api_key="key", edr_key="edr-key") as client:
        assert client.opendata is client.opendata
        assert client.edr is client.edr


def test_module_level_get_timeseries():
    endpoint = f"{EDR}/collections/c1/locations/06260"
    payload = {
        "domain": {"axes": {"t": {"values": ["2026-06-01T00:00:00Z"]}}},
        "parameters": {"TG": {"unit": {"symbol": "C"}, "description": {"en": "t"}}},
        "ranges": {"TG": {"values": [18.5]}},
    }
    with respx.mock(assert_all_called=False) as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(200, json=payload)
        )
        ds = knmidata.get_timeseries(
            "c1",
            "06260",
            start_time="2026-06-01",
            end_time="2026-06-08",
            parameters="TG",
            edr_key="edr-key",
        )
        params = route.calls[0].request.url.params
        assert params["parameter-name"] == "TG"
        assert params["datetime"] == "2026-06-01T00:00:00Z/2026-06-08T00:00:00Z"
        assert params["f"] == "CoverageJSON"
        assert route.calls[0].request.headers["Authorization"] == "edr-key"
    assert ds["TG"].values.tolist() == [18.5]


async def test_edr_only_client_does_not_resolve_opendata_auth():
    # an EDR-only client must not fetch the anonymous Open Data key (network)
    endpoint = f"{EDR}/collections/c1/locations/06260"
    payload = {
        "domain": {"axes": {"t": {"values": ["2026-06-01T00:00:00Z"]}}},
        "parameters": {"TG": {"unit": {"symbol": "C"}, "description": {"en": "t"}}},
        "ranges": {"TG": {"values": [18.5]}},
    }
    with respx.mock(assert_all_called=False) as router:
        docs_route = router.get(ANONYMOUS_KEY_DOCS_URL).mock(
            side_effect=httpx.ConnectError("offline")
        )
        router.get(endpoint).mock(return_value=httpx.Response(200, json=payload))
        async with knmidata.AsyncKNMIClient(edr_key="edr-key") as client:
            ds = await client.get_timeseries("c1", "06260", parameters="TG")
        assert docs_route.call_count == 0
    assert ds["TG"].values.tolist() == [18.5]


def test_version_and_exports():
    assert isinstance(knmidata.__version__, str)
    for name in knmidata.__all__:
        assert hasattr(knmidata, name)
