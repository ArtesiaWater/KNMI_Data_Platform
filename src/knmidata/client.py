"""Composition roots and client facades for asynchronous and synchronous workflows."""

import asyncio
from pathlib import Path
from typing import Any, Self

import xarray as xr

from knmidata._auth import resolve_edr_auth, resolve_open_data_auth
from knmidata._http import AsyncTransport
from knmidata.edr.client import AsyncEDRClient
from knmidata.opendata.catalog import AsyncCatalogClient
from knmidata.opendata.download import AsyncFileDownloader


class AsyncKNMIClient:
    """Asynchronous client composing all KNMI Data Platform API sub-services.

    Parameters
    ----------
    api_key : str | None, default=None
        General API key used for Open Data (or shared application key).
    edr_key : str | None, default=None
        Dedicated EDR key if distinct from the main API key.
    max_concurrency : int, default=15
        Maximum concurrent network connections allowed.
    timeout : float, default=60.0
        Network timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        edr_key: str | None = None,
        max_concurrency: int = 15,
        timeout: float = 60.0,
    ) -> None:
        self._transport = AsyncTransport(
            max_concurrency=max_concurrency, timeout=timeout
        )

        open_data_auth = resolve_open_data_auth(api_key)
        self._catalog = AsyncCatalogClient(
            transport=self._transport, auth=open_data_auth
        )
        self._downloader = AsyncFileDownloader(
            transport=self._transport, auth=open_data_auth
        )

        # Lazy-resolve EDR auth so clients using only Open Data don't error if no EDR key is set
        self._edr_key = edr_key or api_key
        self._edr_client: AsyncEDRClient | None = None

    @property
    def edr(self) -> AsyncEDRClient:
        """Access the EDR client, lazily initializing its authentication."""
        if self._edr_client is None:
            edr_auth = resolve_edr_auth(self._edr_key)
            self._edr_client = AsyncEDRClient(transport=self._transport, auth=edr_auth)
        return self._edr_client

    async def open_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        start_time: str | None = None,
        end_time: str | None = None,
        cache_dir: Path | str | None = None,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """Download matching archive files via Open Data and assemble an xarray.Dataset.

        Parameters
        ----------
        dataset_name : str
            KNMI dataset name.
        dataset_version : str
            Dataset version identifier.
        start_time : str | None, default=None
            ISO-8601 start date/time.
        end_time : str | None, default=None
            ISO-8601 end date/time.
        cache_dir : Path | str | None, default=None
            Local folder where downloaded NetCDF/HDF5 files will be preserved.
        **open_mfdataset_kwargs : Any
            Additional keyword arguments passed to xarray.open_mfdataset.

        Returns
        -------
        xr.Dataset
            Combined multidimensional dataset.
        """
        files = await self._catalog.list_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            start_time=start_time,
            end_time=end_time,
        )
        if not files:
            raise FileNotFoundError(
                f"No files found for dataset '{dataset_name}' (v{dataset_version}) "
                f"between {start_time} and {end_time}."
            )

        downloaded_paths = await self._downloader.download_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filenames=files,
            destination_dir=cache_dir,
        )

        return self._downloader.to_xarray(downloaded_paths, **open_mfdataset_kwargs)

    async def aclose(self) -> None:
        """Close the underlying HTTP transport session."""
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.aclose()


class KNMIClient:
    """Synchronous facade wrapping AsyncKNMIClient for non-async environments.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to `AsyncKNMIClient`.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._async_client = AsyncKNMIClient(**kwargs)

    def open_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        start_time: str | None = None,
        end_time: str | None = None,
        cache_dir: Path | str | None = None,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """Download Open Data files and return an xarray.Dataset synchronously."""
        return asyncio.run(
            self._async_client.open_dataset(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                start_time=start_time,
                end_time=end_time,
                cache_dir=cache_dir,
                **open_mfdataset_kwargs,
            )
        )

    def open_edr_position(
        self,
        collection: str,
        coords: tuple[float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: str = "CoverageJSON",
    ) -> xr.Dataset:
        """Query EDR position endpoint synchronously."""
        return asyncio.run(
            self._async_client.edr.position(
                collection=collection,
                coords=coords,
                datetime=datetime,
                parameters=parameters,
                output_format=output_format,
            )
        )

    def open_edr_area(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        datetime: str | None = None,
        parameters: list[str] | None = None,
        output_format: str = "CoverageJSON",
    ) -> xr.Dataset:
        """Query EDR area endpoint synchronously."""
        return asyncio.run(
            self._async_client.edr.area(
                collection=collection,
                bbox=bbox,
                datetime=datetime,
                parameters=parameters,
                output_format=output_format,
            )
        )

    def close(self) -> None:
        """Close the underlying connection pool synchronously."""
        asyncio.run(self._async_client.aclose())

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()


def open_dataset(
    dataset_name: str,
    dataset_version: str,
    start_time: str | None = None,
    end_time: str | None = None,
    api_key: str | None = None,
    cache_dir: Path | str | None = None,
    **open_mfdataset_kwargs: Any,
) -> xr.Dataset:
    """One-line convenience function to synchronously download and open Open Data files.

    Parameters
    ----------
    dataset_name : str
        KNMI dataset name.
    dataset_version : str
        Dataset version identifier.
    start_time : str | None, default=None
        ISO-8601 start timestamp filter.
    end_time : str | None, default=None
        ISO-8601 end timestamp filter.
    api_key : str | None, default=None
        Optional KNMI API key. Falls back to public open-data key if omitted.
    cache_dir : Path | str | None, default=None
        Directory to store files.
    **open_mfdataset_kwargs : Any
        Forwarded to `xarray.open_mfdataset`.

    Returns
    -------
    xr.Dataset
        Multidimensional dataset.
    """
    with KNMIClient(api_key=api_key) as client:
        return client.open_dataset(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            **open_mfdataset_kwargs,
        )
