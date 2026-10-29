"""Catalog querying and file listing for the KNMI Open Data API."""

import re
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from knmidata._constants import OPENDATA_BASE_URL
from knmidata._exceptions import KNMIError
from knmidata._http import AsyncTransport
from knmidata._time import TimeLike, to_iso8601


@dataclass(slots=True, kw_only=True)
class AsyncCatalogClient:
    """Client for querying file lists in the Open Data catalog.

    Parameters
    ----------
    transport : AsyncTransport
        Shared HTTP transport and concurrency semaphore.
    auth : httpx.Auth
        Authentication strategy for the Open Data gateway.
    """

    transport: AsyncTransport
    auth: httpx.Auth

    async def list_files(
        self,
        dataset_name: str,
        dataset_version: str,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        max_keys: int = 1000,
        sorting: str | None = None,
        order_by: str | None = None,
        file_pattern: str | None = None,
    ) -> list[str]:
        """Fetch matching file names from the catalog, handling pagination automatically.

        Parameters
        ----------
        dataset_name : str
            Name of the KNMI dataset (e.g., 'rad_nl25_rac_mfbs_01h').
        dataset_version : str
            Dataset version identifier (e.g., '2.0').
        start_time : TimeLike | None, default=None
            Start timestamp filter (ISO-8601 string, pandas Timestamp, datetime, or
            date; naive values are assumed UTC). Requires server-side ordering by
            modification time, so `order_by` defaults to 'lastModified' when a time
            filter is given.
        end_time : TimeLike | None, default=None
            End timestamp filter.
        max_keys : int, default=1000
            Number of keys to request per pagination batch (API maximum is 1000).
        sorting : str | None, default=None
            'asc' or 'desc'.
        order_by : str | None, default=None
            Attribute to order/filter by: 'filename' (default), 'lastModified', or
            'created'. Set to 'lastModified' (the default when `start_time` or
            `end_time` is given) to make `begin`/`end` act as ISO-8601 timestamps;
            with the default 'filename' ordering, `begin`/`end` are compared as
            filename strings instead.
        file_pattern : str | None, default=None
            Optional regular expression applied client-side to keep only matching
            file names.

        Returns
        -------
        list[str]
            List of unique file names available for download.
        """
        endpoint = (
            f"{OPENDATA_BASE_URL}/datasets/{dataset_name}/"
            f"versions/{dataset_version}/files"
        )
        begin = to_iso8601(start_time) if start_time is not None else None
        end = to_iso8601(end_time) if end_time is not None else None
        if (
            begin is not None
            and end is not None
            and pd.Timestamp(begin) > pd.Timestamp(end)
        ):
            raise ValueError(f"start_time ({begin}) is after end_time ({end}).")
        if (begin or end) and order_by is None:
            order_by = "lastModified"

        pattern = re.compile(file_pattern) if file_pattern else None
        filenames: list[str] = []
        next_page_token: str | None = None
        seen_tokens: set[str] = set()

        while True:
            params: dict[str, Any] = {"maxKeys": max_keys}
            if next_page_token:
                params["nextPageToken"] = next_page_token
            if sorting:
                params["sorting"] = sorting
            if order_by:
                params["orderBy"] = order_by
            if begin:
                params["begin"] = begin
            if end:
                params["end"] = end

            payload = await self.transport.get_json(
                endpoint, auth=self.auth, params=params
            )

            for item in payload.get("files", []):
                name = item.get("filename")
                if name and (pattern is None or pattern.search(name)):
                    filenames.append(name)

            next_page_token = payload.get("nextPageToken")
            if not next_page_token or not payload.get("isTruncated", False):
                break
            if next_page_token in seen_tokens:
                raise KNMIError(
                    f"Pagination loop detected: nextPageToken {next_page_token!r} "
                    "was already used."
                )
            seen_tokens.add(next_page_token)

        return list(dict.fromkeys(filenames))
