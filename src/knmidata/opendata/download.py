"""Concurrent file downloader and xarray dataset loader for Open Data files."""

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import xarray as xr

from .._http import AsyncTransport

OPENDATA_BASE_URL = "https://api.knmi.nl/open-data/v1"


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
            Temporary AWS S3 presigned download URL.
        """
        endpoint = (
            f"{OPENDATA_BASE_URL}/datasets/{dataset_name}/"
            f"versions/{dataset_version}/files/{filename}/url"
        )
        async with self.transport.semaphore:
            response = await self.transport.client.get(endpoint, auth=self.auth)
            response.raise_for_status()
            return str(response.json()["temporaryDownloadUrl"])

    async def stream_file(self, url: str, target_path: Path) -> Path:
        """Stream download a file from an S3 URL in chunks to bounded memory.

        Parameters
        ----------
        url : str
            Direct presigned download URL.
        target_path : Path
            Destination path on local disk.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        async with self.transport.semaphore and self.transport.client.stream(
            "GET", url
        ) as stream:
            assert stream is not None
            stream.raise_for_status()
            with target_path.open("wb") as f:
                async for chunk in stream.aiter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)

        return target_path

    async def download_files(
        self,
        dataset_name: str,
        dataset_version: str,
        filenames: list[str],
        destination_dir: Path | str | None = None,
    ) -> list[Path]:
        """Download multiple files concurrently using asyncio.gather.

        Parameters
        ----------
        dataset_name : str
            Name of the KNMI dataset.
        dataset_version : str
            Dataset version identifier.
        filenames : list[str]
            List of filenames to download.
        destination_dir : Path | str | None, default=None
            Directory to save files into. Uses a temporary directory if None.

        Returns
        -------
        list[Path]
            Local paths of downloaded files.
        """
        dest = (
            Path(destination_dir)
            if destination_dir
            else Path(tempfile.mkdtemp(prefix="knmi_"))
        )
        dest.mkdir(parents=True, exist_ok=True)

        async def _fetch_and_stream(fname: str) -> Path:
            download_url = await self.get_download_url(
                dataset_name, dataset_version, fname
            )
            return await self.stream_file(download_url, dest / fname)

        tasks = [_fetch_and_stream(fname) for fname in filenames]
        return await asyncio.gather(*tasks)

    @staticmethod
    def to_xarray(
        file_paths: list[Path],
        engine: str = "h5netcdf",
        combine: Literal["by_coords", "nested"] = "by_coords",
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """Assemble local NetCDF/HDF5 files into an xarray.Dataset.

        Parameters
        ----------
        file_paths : list[Path]
            Paths to the downloaded NetCDF files.
        engine : str, default="h5netcdf"
            Xarray backend engine.
        combine : str, default="by_coords"
            Method used by xarray to concatenate datasets.
        **open_mfdataset_kwargs : Any
            Additional options forwarded to `xr.open_mfdataset`.

        Returns
        -------
        xr.Dataset
            Combined multidimensional dataset.
        """
        return xr.open_mfdataset(
            file_paths,
            engine=engine,
            combine=combine,
            **open_mfdataset_kwargs,
        )
