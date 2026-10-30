"""Composition roots and client facades for asynchronous and synchronous workflows."""

import asyncio
import inspect
import tempfile
import threading
from collections.abc import Coroutine
from typing import Any, Self

import xarray as xr

from knmidata._auth import resolve_edr_auth, resolve_open_data_auth
from knmidata._exceptions import NoFilesFoundError
from knmidata._http import AsyncTransport
from knmidata._typing import PathLike, RunCoroutine, T, TimeLike
from knmidata.edr.client import AsyncEDRClient
from knmidata.opendata.catalog import AsyncCatalogClient
from knmidata.opendata.download import AsyncFileDownloader, files_to_dataset


class AsyncKNMIClient:
    """Asynchronous client composing all KNMI Data Platform API sub-services.

    Parameters
    ----------
    api_key : str | None, default=None
        API key used for the Open Data API. Falls back to the KNMI_OPENDATA_API_KEY
        environment variable (or KNMI_OPENDATA_API_KEY_FILE), then to the public
        anonymous key.
    edr_key : str | None, default=None
        Dedicated EDR API key. KNMI issues separate keys per API, so an Open Data
        key does not work for EDR. Falls back to KNMI_EDR_API_KEY (or
        KNMI_EDR_API_KEY_FILE).
    max_concurrency : int, default=15
        Maximum concurrent network requests.
    timeout : float, default=60.0
        Read timeout in seconds.
    max_retries : int, default=3
        Retries for transient failures (429/5xx/transport errors).
    """

    def __init__(
        self,
        api_key: str | None = None,
        edr_key: str | None = None,
        max_concurrency: int = 15,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._transport = AsyncTransport(
            max_concurrency=max_concurrency, timeout=timeout, max_retries=max_retries
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
    def transport(self) -> AsyncTransport:
        """The shared HTTP transport."""
        return self._transport

    @property
    def catalog(self) -> AsyncCatalogClient:
        """The Open Data catalog client."""
        return self._catalog

    @property
    def downloader(self) -> AsyncFileDownloader:
        """The Open Data file downloader."""
        return self._downloader

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
        start_time : str | None, default=None
            Start timestamp (TimeLike: ISO-8601 string, pandas Timestamp,
            datetime, or date; naive values are assumed UTC).
        end_time : str | None, default=None
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
        files = await self._catalog.list_files(
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
                paths = await self._downloader.download_files(
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                    filenames=files,
                    destination_dir=tmp,
                )
                dataset = files_to_dataset(paths, **open_dataset_kwargs)
                dataset.load()
                dataset.close()
                return dataset

        paths = await self._downloader.download_files(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            filenames=files,
            destination_dir=cache_dir,
        )
        return files_to_dataset(paths, **open_dataset_kwargs)

    async def aclose(self) -> None:
        """Close the underlying HTTP transport session."""
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.aclose()


class _SyncFacade:
    """Expose the coroutine methods of an async client as blocking sync calls.

    Parameters
    ----------
    run : RunCoroutine[T]
        Scheduler that runs a coroutine on the background loop and returns its result.
    target : Any
        The async client to wrap.
    """

    def __init__(self, run: RunCoroutine[T], target: Any) -> None:
        self._run = run
        self._target = target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if inspect.iscoroutinefunction(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._run(attr(*args, **kwargs))

            return wrapper
        return attr


class KNMIClient:
    """Synchronous facade wrapping AsyncKNMIClient for non-async environments.

    All coroutines are executed on a dedicated background event loop in a
    separate thread, so the client can be called repeatedly (and from inside
    environments that already run an event loop, such as Jupyter).

    Parameters
    ----------
    api_key : str | None, default=None
        API key used for the Open Data API. Falls back to the KNMI_OPENDATA_API_KEY
        environment variable (or KNMI_OPENDATA_API_KEY_FILE), then to the public
        anonymous key.
    edr_key : str | None, default=None
        Dedicated EDR API key. KNMI issues separate keys per API, so an Open Data
        key does not work for EDR. Falls back to KNMI_EDR_API_KEY (or
        KNMI_EDR_API_KEY_FILE).
    max_concurrency : int, default=15
        Maximum concurrent network requests.
    timeout : float, default=60.0
        Read timeout in seconds.
    max_retries : int, default=3
        Retries for transient failures (429/5xx/transport errors).
    """

    def __init__(
        self,
        api_key: str | None = None,
        edr_key: str | None = None,
        max_concurrency: int = 15,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="knmidata-event-loop", daemon=True
        )
        self._thread.start()
        self._closed = False
        self._async_client = AsyncKNMIClient(
            api_key=api_key,
            edr_key=edr_key,
            max_concurrency=max_concurrency,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the background loop and block until it completes.

        This makes an async client usable in synchronous code, including
        environments that already run an event loop.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def open_dataset(
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
        """Download Open Data files and return an xarray.Dataset synchronously.

        See `AsyncKNMIClient.open_dataset` for parameter descriptions.
        """
        return self._run(
            self._async_client.open_dataset(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                start_time=start_time,
                end_time=end_time,
                cache_dir=cache_dir,
                file_pattern=file_pattern,
                max_files=max_files,
                **open_dataset_kwargs,
            )
        )

    @property
    def catalog(self) -> _SyncFacade:
        """Synchronous facade for the Open Data catalog client."""
        return _SyncFacade(self._run, self._async_client.catalog)

    @property
    def downloader(self) -> _SyncFacade:
        """Synchronous facade for the Open Data file downloader."""
        return _SyncFacade(self._run, self._async_client.downloader)

    @property
    def edr(self) -> _SyncFacade:
        """Synchronous facade for the EDR client."""
        return _SyncFacade(self._run, self._async_client.edr)

    def close(self) -> None:
        """Close the underlying connection pool and stop the background loop."""
        if self._closed:
            return
        self._closed = True
        self._run(self._async_client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()


def open_dataset(
    dataset_name: str,
    dataset_version: str,
    start_time: TimeLike | None = None,
    end_time: TimeLike | None = None,
    api_key: str | None = None,
    cache_dir: PathLike | None = None,
    file_pattern: str | None = None,
    max_files: int | None = None,
    **open_dataset_kwargs: Any,
) -> xr.Dataset:
    """One-line convenience function to synchronously download and open Open Data files.

    Parameters
    ----------
    dataset_name : str
        KNMI dataset name.
    dataset_version : str
        Dataset version identifier.
    start_time : str | None, default=None
        Start timestamp filter (TimeLike: ISO-8601 string, pandas Timestamp,
        datetime, or date; naive values are assumed UTC).
    end_time : str | None, default=None
        End timestamp filter (TimeLike, see start_time).
    api_key : str | None, default=None
        Optional KNMI Open Data API key. Falls back to environment variables and
        the public anonymous key if omitted.
    cache_dir : PathLike | None, default=None
        Directory to store downloaded files. If None, files are removed after
        the dataset is loaded into memory.
    file_pattern : str | None, default=None
        Optional regular expression to keep only matching file names.
    max_files : int | None, default=None
        Maximum number of files to download.
    **open_dataset_kwargs : Any
        Forwarded to `xarray.open_dataset` per file.

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
            file_pattern=file_pattern,
            max_files=max_files,
            **open_dataset_kwargs,
        )
