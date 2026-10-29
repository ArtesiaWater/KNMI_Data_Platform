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
