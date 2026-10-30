"""Composition roots and client facades for asynchronous and synchronous workflows.

The clients expose one namespace per KNMI API, symmetric for async and sync use::

    async with AsyncKNMIClient() as client:
        ds = await client.opendata.open_dataset("dataset", "1.0", ...)
        ts = await client.edr.location_data("collection", "06260", ...)

    with KNMIClient() as client:  # same, but blocking
        ds = client.opendata.open_dataset("dataset", "1.0", ...)
        ts = client.edr.location_data("collection", "06260", ...)
"""

import asyncio
import functools
import inspect
import threading
from collections.abc import Coroutine
from logging import getLogger
from typing import Any, ClassVar, Self

import xarray as xr

from knmidata._auth import resolve_edr_auth, resolve_open_data_auth
from knmidata._http import AsyncTransport
from knmidata._typing import (
    PathLike,
    RunCoroutine,
    T,
    TimeLike,
)
from knmidata.edr.client import AsyncEDRClient
from knmidata.opendata.client import AsyncOpenDataClient

logger = getLogger(__name__)


class AsyncKNMIClient:
    """Asynchronous client composing all KNMI Data Platform API sub-services.

    Attributes
    ----------
    opendata : AsyncOpenDataClient
        Open Data API namespace: `list_files`, `download_files`, `open_dataset`, ...
    edr : AsyncEDRClient
        EDR API namespace: `collections`, `locations`, `location_data`, `position`,
        `area`, `cube`, ... (authentication resolved lazily on first access).

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
        self._opendata = AsyncOpenDataClient(
            transport=self._transport, auth=open_data_auth
        )

        # Lazy-resolve EDR auth so clients using only Open Data don't error if no EDR key is set
        self._edr_key = edr_key
        self._edr_client: AsyncEDRClient | None = None

    @property
    def transport(self) -> AsyncTransport:
        """The shared HTTP transport."""
        return self._transport

    @property
    def opendata(self) -> AsyncOpenDataClient:
        """The Open Data API namespace."""
        return self._opendata

    @property
    def edr(self) -> AsyncEDRClient:
        """The EDR API namespace, lazily initializing its authentication."""
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
        """Shortcut for `client.opendata.open_dataset(...)`."""
        return await self._opendata.open_dataset(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            file_pattern=file_pattern,
            max_files=max_files,
            **open_dataset_kwargs,
        )

    def __repr__(self) -> str:
        """Repr showing the sub-clients (never any API keys)."""
        return (
            f"{type(self).__name__}(opendata={self._opendata!r}, "
            f"edr={self._edr_client!r})"
        )

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

    Coroutine methods are bound as real attributes at construction time (with
    `functools.wraps`, so `__doc__`/`__name__`/`__wrapped__` and the signature are
    preserved), which makes runtime tab-completion (`dir()`) and `help()` work.
    Nested sub-clients (e.g. `client.edr`) are wrapped recursively.

    Parameters
    ----------
    run : RunCoroutine[T]
        Scheduler that runs a coroutine on the background loop and returns its result.
    target : Any
        The async client to wrap.
    """

    _FACADE_TARGETS: ClassVar[tuple[type, ...]] = (
        AsyncEDRClient,
        AsyncOpenDataClient,
    )
    """Async client classes that get wrapped recursively as nested facades."""

    def __repr__(self) -> str:
        """Repr showing the wrapped async client class (no credentials)."""
        return f"{type(self).__name__}(target={type(self._target).__name__})"

    def __init__(self, run: RunCoroutine[T], target: Any) -> None:
        self._run = run
        self._target = target
        for name in dir(target):
            if name.startswith("_"):
                continue
            attr = getattr(target, name)
            if inspect.iscoroutinefunction(attr):
                setattr(self, name, self._wrap(attr))
            elif isinstance(attr, self._FACADE_TARGETS):
                setattr(self, name, _SyncFacade(run, attr))

    def _wrap(self, func: Any) -> Any:
        """Wrap a coroutine function in a blocking call, preserving its metadata."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._run(func(*args, **kwargs))

        return wrapper

    def __getattr__(self, name: str) -> Any:
        # Fallback for attributes not bound in __init__ (e.g. added later).
        attr = getattr(self._target, name)
        if inspect.iscoroutinefunction(attr):
            wrapper = self._wrap(attr)
            setattr(self, name, wrapper)
            return wrapper
        if isinstance(attr, self._FACADE_TARGETS):
            facade = _SyncFacade(self._run, attr)
            setattr(self, name, facade)
            return facade
        return attr

    def __dir__(self) -> list[str]:
        """Include the wrapped target's public attributes for tab-completion."""
        own = {n for n in super().__dir__() if not n.startswith("_")}
        target_names = {n for n in dir(self._target) if not n.startswith("_")}
        return sorted(own | target_names)


class KNMIClient:
    """Synchronous facade wrapping AsyncKNMIClient for non-async environments.

    All coroutines are executed on a dedicated background event loop in a
    separate thread, so the client can be called repeatedly (and from inside
    environments that already run an event loop, such as Jupyter).

    Attributes
    ----------
    opendata : _SyncFacade
        Synchronous facade for the Open Data API namespace.
    edr : _SyncFacade
        Synchronous facade for the EDR API namespace.

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
        logger.debug("KNMIClient: background event loop started")
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

    @property
    def opendata(self) -> _SyncFacade:
        """Synchronous facade for the Open Data API namespace."""
        return _SyncFacade(self._run, self._async_client.opendata)

    @property
    def edr(self) -> _SyncFacade:
        """Synchronous facade for the EDR API namespace."""
        return _SyncFacade(self._run, self._async_client.edr)

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
        """Shortcut for `client.opendata.open_dataset(...)` (synchronous)."""
        return self._run(
            self._async_client.opendata.open_dataset(
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

    def __repr__(self) -> str:
        """Repr showing the wrapped async client and closed state."""
        return f"{type(self).__name__}(client={self._async_client!r}, closed={self._closed})"

    def close(self) -> None:
        """Close the underlying connection pool and stop the background loop."""
        if self._closed:
            return
        self._closed = True
        self._run(self._async_client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)
        logger.debug("KNMIClient: closed (transport closed, background loop stopped)")

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
    start_time : TimeLike | None, default=None
        Start timestamp filter (ISO-8601 string, pandas Timestamp, datetime, or
        date; naive values are assumed UTC).
    end_time : TimeLike | None, default=None
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
        return client.opendata.open_dataset(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            file_pattern=file_pattern,
            max_files=max_files,
            **open_dataset_kwargs,
        )
