"""Open Data API client: catalog listing, file downloads, and dataset assembly."""

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import xarray as xr

from knmidata._exceptions import KNMIError, NoFilesFoundError
from knmidata._http import AsyncTransport
from knmidata._time import to_iso8601
from knmidata._typing import PathLike, QueryParams, TimeLike
from knmidata.config import OPENDATA_BASE_URL
from knmidata.opendata.parsers import files_to_dataset


@dataclass(slots=True, kw_only=True)
class AsyncOpenDataClient:
    """Client for the KNMI Open Data API (file-based bulk downloads).

    Covers the whole Open Data workflow: `list_files` (catalog with cursor
    pagination and time filtering), `get_download_url` (presigned S3 URL per
    file), `stream_file`/`download_files` (atomic concurrent downloads), and
    `open_dataset` (list + download + assemble into an `xarray.Dataset` in one
    call). Access it via `client.opendata`, or instantiate directly with your
    own transport/auth.

    Parameters
    ----------
    transport : AsyncTransport
        Shared HTTP transport and concurrency semaphore.
    auth : httpx.Auth
        Authentication strategy for the Open Data gateway.
    """

    transport: AsyncTransport
    auth: httpx.Auth

    def __repr__(self) -> str:
        """Concise repr showing transport and auth (auth masks any API key)."""
        return (
            f"{type(self).__name__}(transport={self.transport!r}, auth={self.auth!r})"
        )

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
            params: QueryParams = {"maxKeys": max_keys}
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

    async def _get_file_payload(
        self, dataset_name: str, dataset_version: str, filename: str
    ) -> dict[str, Any]:
        """Fetch the raw FileDownload response (metadata + presigned URL) for a file."""
        endpoint = (
            f"{OPENDATA_BASE_URL}/datasets/{dataset_name}/"
            f"versions/{dataset_version}/files/{filename}/url"
        )
        return await self.transport.get_json(endpoint, auth=self.auth)

    async def get_download_url(
        self, dataset_name: str, dataset_version: str, filename: str
    ) -> str:
        """Retrieve a temporary S3 presigned download URL for a file.

        Parameters
        ----------
        dataset_name : str
            Name of the KNMI dataset.
        dataset_version : str
            Dataset version identifier.
        filename : str
            File name inside the dataset.

        Returns
        -------
        str
            Temporary presigned download URL.
        """
        payload = await self._get_file_payload(dataset_name, dataset_version, filename)
        return str(payload["temporaryDownloadUrl"])

    async def stream_file(
        self, url: str, target_path: Path, expected_size: int | None = None
    ) -> Path:
        """Stream download a file from a presigned URL in bounded memory.

        The file is written to a ``.part`` sibling and atomically moved into
        place, so an interrupted download is never mistaken for a complete file.
        Existing complete files are skipped.

        Parameters
        ----------
        url : str
            Direct presigned download URL.
        target_path : Path
            Destination path on local disk.
        expected_size : int | None, default=None
            Expected file size in bytes; used to validate the download and to
            decide whether an existing file can be skipped.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        if target_path.exists():
            existing = target_path.stat().st_size
            if existing > 0 and (expected_size is None or existing == expected_size):
                return target_path

        part_path = target_path.with_name(target_path.name + ".part")
        async with self.transport.stream("GET", url) as stream:
            with part_path.open("wb") as f:
                async for chunk in stream.aiter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)

        downloaded = part_path.stat().st_size
        if expected_size is not None and downloaded != expected_size:
            part_path.unlink(missing_ok=True)
            raise KNMIError(
                f"Downloaded size {downloaded} does not match expected size "
                f"{expected_size} for {target_path.name}."
            )
        os.replace(part_path, target_path)
        return target_path

    async def download_files(
        self,
        dataset_name: str,
        dataset_version: str,
        filenames: list[str],
        destination_dir: PathLike | None = None,
    ) -> list[Path]:
        """Download multiple files concurrently.

        Parameters
        ----------
        dataset_name : str
            Name of the KNMI dataset.
        dataset_version : str
            Dataset version identifier.
        filenames : list[str]
            List of filenames to download.
        destination_dir : PathLike | None, default=None
            Directory to save files into. Uses a temporary directory if None.

        Returns
        -------
        list[Path]
            Local paths of downloaded files, in the same order as `filenames`.

        Raises
        ------
        KNMIError
            If one or more files could not be downloaded.
        """
        dest = (
            Path(destination_dir)
            if destination_dir
            else Path(tempfile.mkdtemp(prefix="knmi_"))
        )
        dest.mkdir(parents=True, exist_ok=True)

        async def _fetch_and_stream(fname: str) -> Path:
            payload = await self._get_file_payload(dataset_name, dataset_version, fname)
            return await self.stream_file(
                str(payload["temporaryDownloadUrl"]),
                dest / Path(fname).name,
                int(payload["size"]),
            )

        tasks = [_fetch_and_stream(fname) for fname in filenames]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [
            (fname, result)
            for fname, result in zip(filenames, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if errors:
            details = "; ".join(f"{fname}: {err}" for fname, err in errors)
            raise KNMIError(
                f"Failed to download {len(errors)} of {len(filenames)} files. {details}"
            )
        return [result for result in results if isinstance(result, Path)]

    async def open_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        cache_dir: PathLike | None = None,
        file_pattern: str | None = None,
        max_files: int | None = None,
        **open_dataset_kwargs: Any,
    ) -> xr.Dataset:
        """Download matching archive files via Open Data and assemble an xarray.Dataset.

        Parameters
        ----------
        dataset_name : str
            KNMI dataset name.
        dataset_version : str
            Dataset version identifier.
        start_time : TimeLike | None, default=None
            Start timestamp (ISO-8601 string, pandas Timestamp, datetime, or
            date; naive values are assumed UTC).
        end_time : TimeLike | None, default=None
            End timestamp (TimeLike, see start_time).
        cache_dir : PathLike | None, default=None
            Local folder where downloaded NetCDF/HDF5 files are preserved and the
            dataset is opened lazily. If None, files are downloaded to a temporary
            directory, loaded into memory, and removed again.
        file_pattern : str | None, default=None
            Optional regular expression to keep only matching file names.
        max_files : int | None, default=None
            Maximum number of files to download (after listing, ascending order).
        **open_dataset_kwargs : Any
            Additional keyword arguments passed to `xarray.open_dataset` per file.

        Returns
        -------
        xr.Dataset
            Combined multidimensional dataset.

        Raises
        ------
        NoFilesFoundError
            If the query matches no files.
        """
        files = await self.list_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            start_time=start_time,
            end_time=end_time,
            file_pattern=file_pattern,
        )
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise NoFilesFoundError(
                f"No files found for dataset '{dataset_name}' (v{dataset_version}) "
                f"between {start_time} and {end_time}."
            )

        if cache_dir is None:
            with tempfile.TemporaryDirectory(prefix="knmi_") as tmp:
                paths = await self.download_files(
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                    filenames=files,
                    destination_dir=tmp,
                )
                dataset = files_to_dataset(paths, **open_dataset_kwargs)
                dataset.load()
                dataset.close()
                return dataset

        paths = await self.download_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filenames=files,
            destination_dir=cache_dir,
        )
        return files_to_dataset(paths, **open_dataset_kwargs)
