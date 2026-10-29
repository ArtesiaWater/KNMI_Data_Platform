"""Asynchronous Python client for the KNMI Data Platform with native xarray support."""

# ruff: noqa: F401
import logging

from knmidata._auth import APIKeyAuth, NoAuth, resolve_edr_auth, resolve_open_data_auth
from knmidata.client import AsyncKNMIClient, KNMIClient, open_dataset

from ._version import __version__, show_versions

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = [
    "APIKeyAuth",
    "AsyncKNMIClient",
    "KNMIClient",
    "NoAuth",
    "open_dataset",
    "resolve_edr_auth",
    "resolve_open_data_auth",
]
