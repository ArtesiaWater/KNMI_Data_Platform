"""Tests for sensible __repr__ implementations (and that they never leak secrets)."""

import knmidata
from knmidata._auth import APIKeyAuth, NoAuth
from knmidata._http import AsyncTransport
from knmidata.opendata.download import FileDownloadInfo

SECRET_KEY = "eyJ" + "s3cr3t" * 20
PRESIGNED = "https://fake-s3.amazonaws.com/f.nc?X-Amz-Signature=abc123"


def test_api_key_auth_repr_masks_key():
    auth = APIKeyAuth(api_key=SECRET_KEY)
    r = repr(auth)
    assert "APIKeyAuth" in r
    assert SECRET_KEY not in r
    assert "***" in r


def test_no_auth_repr():
    assert "NoAuth" in repr(NoAuth())


def test_file_download_info_repr_masks_url():
    info = FileDownloadInfo(
        content_type="application/x-netcdf",
        size=10,
        last_modified="2026-06-01",
        url=PRESIGNED,
    )
    r = repr(info)
    assert "FileDownloadInfo" in r
    assert "size=10" in r
    assert PRESIGNED not in r


def test_transport_repr():
    t = AsyncTransport(max_concurrency=3, timeout=5.0, max_retries=1)
    r = repr(t)
    assert r == "AsyncTransport(max_concurrency=3, timeout=5.0, max_retries=1)"


def test_client_reprs_do_not_leak_keys():
    with knmidata.KNMIClient(api_key=SECRET_KEY, edr_key=SECRET_KEY) as client:
        r = repr(client)
        assert "KNMIClient" in r
        assert SECRET_KEY not in r
        r_opendata = repr(client.opendata)
        assert "_SyncFacade" in r_opendata
        assert SECRET_KEY not in r_opendata
        # the underlying auth object repr is masked too
        assert SECRET_KEY not in repr(client.opendata._target.auth)
    import asyncio

    async def check_async():
        async with knmidata.AsyncKNMIClient(
            api_key=SECRET_KEY, edr_key=SECRET_KEY
        ) as ac:
            r = repr(ac)
            assert "AsyncKNMIClient" in r
            assert SECRET_KEY not in r
            assert SECRET_KEY not in repr(ac.opendata)

    asyncio.run(check_async())
