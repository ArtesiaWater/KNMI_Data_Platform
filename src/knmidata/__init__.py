"""Asynchronous Python client for the KNMI Data Platform with native xarray support."""

import logging

from knmidata._auth import APIKeyAuth, NoAuth, resolve_edr_auth, resolve_open_data_auth
from knmidata._exceptions import (
    KNMIAuthError,
    KNMIError,
    KNMIHTTPError,
    NoFilesFoundError,
)
from knmidata._version import __version__, show_versions
from knmidata.client import AsyncKNMIClient, KNMIClient, open_dataset

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "APIKeyAuth",
    "AsyncKNMIClient",
    "KNMIAuthError",
    "KNMIClient",
    "KNMIError",
    "KNMIHTTPError",
    "NoAuth",
    "NoFilesFoundError",
    "__version__",
    "open_dataset",
    "resolve_edr_auth",
    "resolve_open_data_auth",
    "show_versions",
]
