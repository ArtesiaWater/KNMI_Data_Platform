"""Tests for the Open Data file downloader and dataset assembly."""

import httpx
import pandas as pd
import pytest
import respx
import xarray as xr

from knmidata._auth import APIKeyAuth
from knmidata._exceptions import KNMIError
from knmidata.config import OPENDATA_BASE_URL as OPENDATA
from knmidata.opendata.download import AsyncFileDownloader, files_to_dataset

S3 = "https://fake-s3.amazonaws.com"


@pytest.fixture
def downloader(transport):
    return AsyncFileDownloader(transport=transport, auth=APIKeyAuth(api_key="key"))


def _url_payload(filename, size):
    return {
        "contentType": "application/x-netcdf",
        "size": str(size),
        "lastModified": "2024-01-01T00:00:00Z",
        "temporaryDownloadUrl": f"{S3}/{filename}",
    }


async def test_get_download_info(downloader):
    url = f"{OPENDATA}/datasets/ds/versions/1.0/files/f1.nc/url"
    with respx.mock(assert_all_called=False) as router:
        router.get(url).mock(
            return_value=httpx.Response(200, json=_url_payload("f1.nc", 123))
        )
        info = await downloader.get_download_info("ds", "1.0", "f1.nc")
        assert info.url == f"{S3}/f1.nc"
        assert info.size == 123
        assert info.content_type == "application/x-netcdf"


async def test_download_files_streams_and_skips_existing(
    downloader, netcdf_bytes, tmp_path
):
    url_endpoint = f"{OPENDATA}/datasets/ds/versions/1.0/files/f1.nc/url"
    with respx.mock(assert_all_called=False) as router:
        router.get(url_endpoint).mock(
            return_value=httpx.Response(
                200, json=_url_payload("f1.nc", len(netcdf_bytes))
            )
        )
        s3_route = router.get(f"{S3}/f1.nc").mock(
            return_value=httpx.Response(200, content=netcdf_bytes)
        )
        out = tmp_path / "out"
        paths = await downloader.download_files(
            "ds", "1.0", ["f1.nc"], destination_dir=out
        )
        assert paths == [out / "f1.nc"]
        assert (out / "f1.nc").read_bytes() == netcdf_bytes
        assert not (out / "f1.nc.part").exists()
        assert s3_route.call_count == 1

        # second call: file exists with matching size, no new S3 request
        await downloader.download_files("ds", "1.0", ["f1.nc"], destination_dir=out)
        assert s3_route.call_count == 1


async def test_stream_file_rejects_size_mismatch(downloader, tmp_path):
    target = tmp_path / "f1.nc"
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{S3}/f1.nc").mock(
            return_value=httpx.Response(200, content=b"short")
        )
        with pytest.raises(KNMIError, match="does not match expected size"):
            await downloader.stream_file(f"{S3}/f1.nc", target, expected_size=100)
    assert not target.exists()
    assert not (tmp_path / "f1.nc.part").exists()


async def test_download_files_aggregates_errors(downloader, netcdf_bytes, tmp_path):
    ok_url = f"{OPENDATA}/datasets/ds/versions/1.0/files/ok.nc/url"
    bad_url = f"{OPENDATA}/datasets/ds/versions/1.0/files/bad.nc/url"
    with respx.mock(assert_all_called=False) as router:
        router.get(ok_url).mock(
            return_value=httpx.Response(
                200, json=_url_payload("ok.nc", len(netcdf_bytes))
            )
        )
        router.get(bad_url).mock(return_value=httpx.Response(500, text="boom"))
        router.get(f"{S3}/ok.nc").mock(
            return_value=httpx.Response(200, content=netcdf_bytes)
        )
        with pytest.raises(KNMIError, match=r"bad\.nc"):
            await downloader.download_files(
                "ds", "1.0", ["ok.nc", "bad.nc"], destination_dir=tmp_path
            )
    # the successful file is still on disk
    assert (tmp_path / "ok.nc").exists()


def test_files_to_dataset_combines_by_coords(tmp_path):
    paths = []
    for i, day in enumerate(["2024-01-01", "2024-01-02"]):
        p = tmp_path / f"f{i}.nc"
        xr.Dataset(
            {"temp": ("time", [float(i)])},
            coords={"time": pd.to_datetime([f"{day}T00:00:00"])},
        ).to_netcdf(p, engine="h5netcdf")
        paths.append(p)
    ds = files_to_dataset(paths)
    assert ds["temp"].dims == ("time",)
    assert ds.sizes["time"] == 2
    assert ds["temp"].values.tolist() == [0.0, 1.0]
