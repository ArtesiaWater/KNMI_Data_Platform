"""Asynchronous Python client for the KNMI Data Platform with native xarray support."""

import logging
import re

from knmidata._auth import APIKeyAuth, NoAuth, resolve_edr_auth, resolve_open_data_auth
from knmidata._exceptions import (
    KNMIAuthError,
    KNMIError,
    KNMIHTTPError,
    NoFilesFoundError,
)
from knmidata._http import AsyncTransport
from knmidata._typing import (
    BBox,
    CombineMethod,
    Coords,
    FilePaths,
    OutputFormat,
    PathLike,
    TimeLike,
    ZLevel,
)
from knmidata._version import __version__, show_versions
from knmidata.client import AsyncKNMIClient, KNMIClient, open_dataset
from knmidata.edr.client import AsyncEDRClient
from knmidata.edr.parsers import parse_coveragejson, parse_geojson
from knmidata.opendata.client import AsyncOpenDataClient
from knmidata.opendata.parsers import files_to_dataset

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "APIKeyAuth",
    "AsyncEDRClient",
    "AsyncKNMIClient",
    "AsyncOpenDataClient",
    "AsyncTransport",
    "BBox",
    "CombineMethod",
    "Coords",
    "FilePaths",
    "KNMIAuthError",
    "KNMIClient",
    "KNMIError",
    "KNMIHTTPError",
    "NoAuth",
    "NoFilesFoundError",
    "OutputFormat",
    "PathLike",
    "TimeLike",
    "ZLevel",
    "__version__",
    "files_to_dataset",
    "open_dataset",
    "parse_coveragejson",
    "parse_geojson",
    "resolve_edr_auth",
    "resolve_open_data_auth",
    "show_versions",
]
