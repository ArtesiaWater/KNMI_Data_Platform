"""Catalog querying and file listing for the KNMI Open Data API."""

from dataclasses import dataclass
from typing import Any

import httpx

from knmidata._http import AsyncTransport

OPENDATA_BASE_URL = "https://api.knmi.nl/open-data/v1"


@dataclass(slots=True, kw_only=True)
class AsyncCatalogClient:
    """Client for querying datasets, versions, and file lists in the Open Data catalog.

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
        start_time: str | None = None,
        end_time: str | None = None,
        max_keys: int = 500,
    ) -> list[str]:
        """Fetch matching file names from the catalog, handling pagination automatically.

        Parameters
        ----------
        dataset_name : str
            Name of the KNMI dataset (e.g., 'rad_nl25_rac_mfbs_01h').
        dataset_version : str
            Dataset version identifier (e.g., '2.0').
        start_time : str | None, default=None
            ISO-8601 start timestamp filter if supported by the dataset.
        end_time : str | None, default=None
            ISO-8601 end timestamp filter if supported by the dataset.
        max_keys : int, default=500
            Number of keys to request per pagination batch.

        Returns
        -------
        list[str]
            List of unique file names available for download.
        """
        endpoint = (
            f"{OPENDATA_BASE_URL}/datasets/{dataset_name}/"
            f"versions/{dataset_version}/files"
        )
        filenames: list[str] = []
        next_page_token: str | None = None

        while True:
            params: dict[str, Any] = {"maxKeys": max_keys}
            if next_page_token:
                params["nextPageToken"] = next_page_token
            if start_time:
                params["begin"] = start_time
            if end_time:
                params["end"] = end_time

            async with self.transport.semaphore:
                response = await self.transport.client.get(
                    endpoint, params=params, auth=self.auth
                )
                response.raise_for_status()
                payload = response.json()

            for item in payload.get("files", []):
                if name := item.get("filename"):
                    filenames.append(name)

            next_page_token = payload.get("nextPageToken")
            if not next_page_token or not payload.get("isTruncated", False):
                break

        return filenames
