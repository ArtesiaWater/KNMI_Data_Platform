"""Open Data API client: catalog listing, file downloads, and dataset assembly."""

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import xarray as xr

from knmidata._exceptions import KNMIError, NoFilesFoundError
from knmidata._http import AsyncTransport
from knmidata._time import to_iso8601
from knmidata._typing import CombineMethod, FilePaths, PathLike, QueryParams, TimeLike
from knmidata.config import OPENDATA_BASE_URL
from knmidata.opendata.parsers import files_to_dataset

logger = getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class AsyncOpenDataClient:
    """Client for the KNMI Open Data API (file-based bulk downloads).

    Covers the whole Open Data workflow: `list_files` (catalog with cursor
    pagination and time filtering), `get_download_url` (presigned S3 URL per
    file), `stream_file`/`download_files` (atomic concurrent downloads to disk),
    `open_dataset` (open files already on disk into an `xarray.Dataset`), and
    `get_dataset` (list + download-if-missing + open in one call). Access it
    via `client.opendata`, or instantiate directly with your own transport/auth.

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
        max_files: int | None = None,
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
            Number of keys to request per pagination batch. Clamped to the API
            maximum of 1000.
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
        max_files : int | None, default=None
            Stop paginating once this many matching files have been collected
            (after client-side pattern filtering).

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
        max_keys = max(1, min(max_keys, 1000))

        pattern = re.compile(file_pattern) if file_pattern else None
        filenames: list[str] = []
        next_page_token: str | None = None
        seen_tokens: set[str] = set()

        logger.info(
            "Listing files of dataset '%s' v%s, %s to %s (order_by=%s)",
            dataset_name,
            dataset_version,
            begin or "..",
            end or "..",
            order_by or "filename",
        )
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
            logger.debug(
                "Catalog page: %d files (isTruncated=%s, nextPageToken=%s)",
                len(payload.get("files", [])),
                payload.get("isTruncated"),
                bool(payload.get("nextPageToken")),
            )

            for item in payload.get("files", []):
                name = item.get("filename")
                if name and (pattern is None or pattern.search(name)):
                    filenames.append(name)

            if max_files is not None and len(filenames) >= max_files:
                break
            next_page_token = payload.get("nextPageToken")
            if not next_page_token or not payload.get("isTruncated", False):
                break
            if next_page_token in seen_tokens:
                raise KNMIError(
                    f"Pagination loop detected: nextPageToken {next_page_token!r} "
                    "was already used."
                )
            seen_tokens.add(next_page_token)

        result = list(dict.fromkeys(filenames))
        if max_files is not None:
            result = result[:max_files]
        logger.info(
            "Found %d unique files for dataset '%s' v%s",
            len(result),
            dataset_name,
            dataset_version,
        )
        return result

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
                logger.debug(
                    "Skipping existing file %s (%d bytes)", target_path, existing
                )
                return target_path

        logger.debug("Downloading %s -> %s", url.split("?")[0], target_path)
        part_path = target_path.with_name(target_path.name + ".part")
        async with self.transport.stream("GET", url) as stream:
            with part_path.open("wb") as f:
                async for chunk in stream.aiter_bytes(chunk_size=1024 * 64):
                    # keep blocking disk I/O off the event loop
                    await asyncio.to_thread(f.write, chunk)

        downloaded = part_path.stat().st_size
        if expected_size is not None and downloaded != expected_size:
            logger.warning(
                "Size mismatch for %s: downloaded %d bytes, expected %d — discarding partial file",
                target_path.name,
                downloaded,
                expected_size,
            )
            part_path.unlink(missing_ok=True)
            raise KNMIError(
                f"Downloaded size {downloaded} does not match expected size "
                f"{expected_size} for {target_path.name}."
            )
        os.replace(part_path, target_path)
        logger.debug("Downloaded %s (%d bytes)", target_path.name, downloaded)
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
        logger.info(
            "Downloading %d files of dataset '%s' v%s to %s",
            len(filenames),
            dataset_name,
            dataset_version,
            dest,
        )

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
            logger.error(
                "Failed to download %d of %d files of dataset '%s' v%s: %s",
                len(errors),
                len(filenames),
                dataset_name,
                dataset_version,
                details,
            )
            raise KNMIError(
                f"Failed to download {len(errors)} of {len(filenames)} files. {details}"
            )
        logger.info("Downloaded %d files to %s", len(filenames), dest)
        return [result for result in results if isinstance(result, Path)]

    async def open_dataset(
        self,
        source: PathLike | FilePaths,
        engine: str = "h5netcdf",
        combine: CombineMethod = "by_coords",
        concat_dim: str | None = None,
        **open_dataset_kwargs: Any,
    ) -> xr.Dataset:
        """Open already-downloaded files from disk into an xarray.Dataset (no network).

        Parameters
        ----------
        source : PathLike | FilePaths
            A directory (all files in it, sorted by name — e.g. a `cache_dir`
            previously filled by `get_dataset`) or an explicit list of file paths.
        engine : str, default="h5netcdf"
            Xarray backend engine used to open each file (see `files_to_dataset`).
        combine : CombineMethod, default="by_coords"
            How to combine the per-file datasets (see `files_to_dataset`).
        concat_dim : str | None, default=None
            Dimension to concatenate along when combine='nested'.
        **open_dataset_kwargs : Any
            Additional keyword arguments passed to `xarray.open_dataset` per file.

        Returns
        -------
        xr.Dataset
            Combined multidimensional dataset, opened lazily from disk.
        """
        if isinstance(source, str | Path):
            paths = sorted(p for p in Path(source).iterdir() if p.is_file())
        else:
            paths = list(source)
        return await asyncio.to_thread(
            files_to_dataset,
            paths,
            engine=engine,
            combine=combine,
            concat_dim=concat_dim,
            **open_dataset_kwargs,
        )

    async def get_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        start_time: TimeLike | None = None,
        end_time: TimeLike | None = None,
        cache_dir: PathLike | None = None,
        file_pattern: str | None = None,
        max_files: int | None = None,
        engine: str = "h5netcdf",
        combine: CombineMethod = "by_coords",
        concat_dim: str | None = None,
        **open_dataset_kwargs: Any,
    ) -> xr.Dataset:
        """Get matching archive files via Open Data and assemble an xarray.Dataset.

        If `cache_dir` is given, files already on disk (matching the catalog size)
        are reused instead of re-downloaded, and the dataset is opened lazily from
        disk via `open_dataset`. If `cache_dir` is None, files are downloaded to a
        temporary directory, loaded into memory, and removed again.

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
            Local folder to cache downloaded NetCDF/HDF5 files in. If None, files
            are downloaded to a temporary directory and loaded into memory.
        file_pattern : str | None, default=None
            Optional regular expression to keep only matching file names.
        max_files : int | None, default=None
            Maximum number of files to download (after listing, ascending order).
        engine : str, default="h5netcdf"
            Xarray backend engine used to open each file (see `files_to_dataset`).
        combine : CombineMethod, default="by_coords"
            How to combine the per-file datasets (see `files_to_dataset`).
        concat_dim : str | None, default=None
            Dimension to concatenate along when combine='nested'.
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
            max_files=max_files,
        )
        logger.info(
            "get_dataset: dataset '%s' v%s -> %d files (cache_dir=%s)",
            dataset_name,
            dataset_version,
            len(files),
            cache_dir or "<temporary, loaded into memory>",
        )
        if not files:
            raise NoFilesFoundError(
                f"No files found for dataset '{dataset_name}' (v{dataset_version}) "
                f"between {start_time} and {end_time}."
            )

        combine_kwargs = {
            "engine": engine,
            "combine": combine,
            "concat_dim": concat_dim,
            **open_dataset_kwargs,
        }
        if cache_dir is None:
            with tempfile.TemporaryDirectory(prefix="knmi_") as tmp:
                paths = await self.download_files(
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                    filenames=files,
                    destination_dir=tmp,
                )
                dataset = await self.open_dataset(paths, **combine_kwargs)
                dataset.load()
                dataset.close()
                return dataset

        paths = await self.download_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filenames=files,
            destination_dir=cache_dir,
        )
        return await self.open_dataset(paths, **combine_kwargs)
