"""Tests for version reporting and the public API surface."""

import knmidata
from knmidata._version import get_versions, show_versions


def test_get_versions():
    versions = get_versions()
    assert versions["knmidata"] == knmidata.__version__
    assert versions["httpx"]
    assert versions["xarray"]


def test_show_versions(capsys):
    show_versions()
    out = capsys.readouterr().out
    assert "knmidata" in out
    assert "httpx" in out


def test_subclients_exposed_at_namespace():
    # users can instantiate the sub-clients directly with their own transport/auth
    for name in (
        "AsyncEDRClient",
        "AsyncOpenDataClient",
        "AsyncTransport",
    ):
        assert name in knmidata.__all__
        assert isinstance(getattr(knmidata, name), type)
    transport = knmidata.AsyncTransport(max_concurrency=2)
    auth = knmidata.APIKeyAuth(api_key="k")
    edr = knmidata.AsyncEDRClient(transport=transport, auth=auth)
    opendata = knmidata.AsyncOpenDataClient(transport=transport, auth=auth)
    assert isinstance(edr, knmidata.AsyncEDRClient)
    assert isinstance(opendata, knmidata.AsyncOpenDataClient)
    # all Open Data methods live directly on AsyncOpenDataClient now
    for method in (
        "list_files",
        "get_download_info",
        "stream_file",
        "download_files",
        "open_dataset",
    ):
        assert hasattr(opendata, method)
