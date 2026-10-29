"""Concurrent file downloader and xarray dataset loader for Open Data files."""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import xarray as xr

from knmidata._constants import OPENDATA_BASE_URL
from knmidata._exceptions import KNMIError
from knmidata._http import AsyncTransport
from knmidata._typing import CombineMethod, FilePaths, PathLike


@dataclass(slots=True, kw_only=True)
class FileDownloadInfo:
    """Metadata and presigned URL for a single dataset file.

    Parameters
    ----------
    content_type : str
        MIME type of the file.
    size : int
        File size in bytes, as reported by the API.
    last_modified : str
        ISO-8601 timestamp of the last modification.
    url : str
        Temporary presigned download URL.
    """

    content_type: str
    size: int
    last_modified: str
    url: str


@dataclass(slots=True, kw_only=True)
class AsyncFileDownloader:
    """Manages S3 presigned URL retrieval, streaming downloads, and xarray assembly.

    Parameters
    ----------
    transport : AsyncTransport
        Shared HTTP transport and concurrency semaphore.
    auth : httpx.Auth
        Authentication strategy for the Open Data gateway.
    """

    transport: AsyncTransport
    auth: httpx.Auth

    async def get_download_info(
        self, dataset_name: str, dataset_version: str, filename: str
    ) -> FileDownloadInfo:
        """Retrieve metadata and a temporary S3 presigned URL for a file.

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
        FileDownloadInfo
            File metadata including the presigned download URL.
        """
        endpoint = (
            f"{OPENDATA_BASE_URL}/datasets/{dataset_name}/"
            f"versions/{dataset_version}/files/{filename}/url"
        )
        payload = await self.transport.get_json(endpoint, auth=self.auth)
        return FileDownloadInfo(
            content_type=str(payload["contentType"]),
            size=int(payload["size"]),
            last_modified=str(payload["lastModified"]),
            url=str(payload["temporaryDownloadUrl"]),
        )

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
            info = await self.get_download_info(dataset_name, dataset_version, fname)
            return await self.stream_file(info.url, dest / Path(fname).name, info.size)

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


def files_to_dataset(
    file_paths: FilePaths,
    engine: str = "h5netcdf",
    combine: CombineMethod = "by_coords",
    concat_dim: str | None = None,
    **open_dataset_kwargs: Any,
) -> xr.Dataset:
    """Assemble local NetCDF/HDF5 files into an xarray.Dataset.

    Implemented with `xr.open_dataset` + `xr.combine_by_coords` so that no dask
    installation is required.

    Parameters
    ----------
    file_paths : FilePaths
        Paths to the downloaded NetCDF files.
    engine : str, default="h5netcdf"
        Xarray backend engine.
    combine : str, default="by_coords"
        How to combine the per-file datasets: 'by_coords' (combine_by_coords),
        'nested' (concat along `concat_dim`), or 'no_conflicts' (merge).
    concat_dim : str | None, default=None
        Dimension to concatenate along when combine='nested'.
    **open_dataset_kwargs : Any
        Additional options forwarded to `xr.open_dataset`.

    Returns
    -------
    xr.Dataset
        Combined multidimensional dataset.

    Raises
    ------
    KNMIError
        If `file_paths` is empty, or combine='nested' without `concat_dim`.
    """
    paths = [str(p) for p in file_paths]
    if not paths:
        raise KNMIError("files_to_dataset() requires at least one file path.")
    datasets = [xr.open_dataset(p, engine=engine, **open_dataset_kwargs) for p in paths]
    if combine == "by_coords":
        return cast("xr.Dataset", xr.combine_by_coords(datasets))
    if combine == "no_conflicts":
        return xr.merge(datasets, compat="no_conflicts")
    if concat_dim is None:
        raise KNMIError("combine='nested' requires concat_dim to be set.")
    return xr.concat(datasets, dim=concat_dim)
