"""Open Data API client: catalog listing, file downloads, and dataset assembly."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import xarray as xr

from knmidata._exceptions import NoFilesFoundError
from knmidata._http import AsyncTransport
from knmidata._typing import PathLike, TimeLike
from knmidata.opendata.catalog import AsyncCatalogClient
from knmidata.opendata.download import (
    AsyncFileDownloader,
    FileDownloadInfo,
    files_to_dataset,
)


@dataclass(slots=True, kw_only=True)
class AsyncOpenDataClient:
    """Client for the KNMI Open Data API (file-based bulk downloads).

    Composes the catalog (file listing) and downloader (presigned URLs +
    streaming) and adds `open_dataset` to go from dataset name to
    `xarray.Dataset` in one call. Access it via `client.opendata`.

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

    _catalog: AsyncCatalogClient = field(init=False)
    _downloader: AsyncFileDownloader = field(init=False)

    def __post_init__(self) -> None:
        """Create the catalog and downloader sub-clients."""
        self._catalog = AsyncCatalogClient(transport=self.transport, auth=self.auth)
        self._downloader = AsyncFileDownloader(transport=self.transport, auth=self.auth)

    @property
    def catalog(self) -> AsyncCatalogClient:
        """The catalog client (file listing)."""
        return self._catalog

    @property
    def downloader(self) -> AsyncFileDownloader:
        """The file downloader (presigned URLs + streaming)."""
        return self._downloader

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
        """List files in a dataset. See `AsyncCatalogClient.list_files`."""
        return await self._catalog.list_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            start_time=start_time,
            end_time=end_time,
            max_keys=max_keys,
            sorting=sorting,
            order_by=order_by,
            file_pattern=file_pattern,
        )

    async def get_download_info(
        self, dataset_name: str, dataset_version: str, filename: str
    ) -> FileDownloadInfo:
        """Get metadata and a presigned URL for one file. See `AsyncFileDownloader.get_download_info`."""
        return await self._downloader.get_download_info(
            dataset_name, dataset_version, filename
        )

    async def download_files(
        self,
        dataset_name: str,
        dataset_version: str,
        filenames: list[str],
        destination_dir: PathLike | None = None,
    ) -> list[Path]:
        """Download files concurrently. See `AsyncFileDownloader.download_files`."""
        return await self._downloader.download_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filenames=filenames,
            destination_dir=destination_dir,
        )

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
