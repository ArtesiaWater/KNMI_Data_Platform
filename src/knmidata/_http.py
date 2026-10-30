"""HTTP transport layer with concurrency control, retries, and error mapping."""

import asyncio
import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Any, Self

import httpx

from knmidata._exceptions import KNMIAuthError, KNMIHTTPError, api_error_message
from knmidata._version import __version__

logger = getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class AsyncTransport:
    """Encapsulates an httpx.AsyncClient bounded by a concurrency semaphore.

    All requests should go through :meth:`request` (or :meth:`stream`), which
    applies the semaphore, retries transient failures with backoff, and maps
    HTTP errors onto the knmidata exception hierarchy.

    Parameters
    ----------
    max_concurrency : int, default=15
        Maximum number of concurrent network requests permitted across the session.
    timeout : float, default=60.0
        Read timeout in seconds (per chunk for streamed downloads).
    max_retries : int, default=3
        Number of retries for transient failures (429/5xx/transport errors).
    retry_backoff : float, default=0.5
        Base backoff in seconds; the actual delay is ``retry_backoff * 2**attempt``
        plus jitter, and honors the ``Retry-After`` response header when present.
    """

    def __init__(
        self,
        max_concurrency: int = 15,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency,
            ),
            headers={"User-Agent": f"knmidata/{__version__}"},
        )

    @staticmethod
    def _response_text(response: httpx.Response) -> str:
        """Return the response body, or an empty string for unread streams."""
        try:
            return response.text
        except httpx.ResponseNotRead:
            return ""

    @classmethod
    def _raise_for_status(cls, response: httpx.Response) -> None:
        """Map an HTTP error response onto the knmidata exception hierarchy."""
        if response.status_code < 400:
            return
        body = cls._response_text(response)
        try:
            message = api_error_message(response.json())
        except (ValueError, httpx.ResponseNotRead):
            message = ""
        detail = message or body
        if response.status_code in (401, 403):
            raise KNMIAuthError(
                f"KNMI API rejected the API key for {response.url} "
                f"(HTTP {response.status_code}): {detail[:500]}. "
                "Request a key at https://developer.dataplatform.knmi.nl/ and pass it "
                "via `api_key`/`edr_key` or the KNMI_API_KEY / KNMI_EDR_API_KEY "
                "environment variables."
            )
        if response.status_code >= 400:
            raise KNMIHTTPError(
                status_code=response.status_code,
                url=str(response.url),
                body=detail,
            )

    async def _backoff_delay(
        self, attempt: int, response: httpx.Response | None
    ) -> float:
        """Compute the retry delay, honoring Retry-After when present."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        jitter = random.uniform(0, 0.1 * self.retry_backoff)
        return self.retry_backoff * (2**attempt) + jitter

    async def request(
        self, method: str, url: str, *, auth: httpx.Auth | None = None, **kwargs: Any
    ) -> httpx.Response:
        """Perform an HTTP request with semaphore, retries, and error mapping.

        Parameters
        ----------
        method : str
            HTTP method (e.g. 'GET').
        url : str
            Absolute URL.
        auth : httpx.Auth | None, default=None
            Authentication strategy; pass None for unauthenticated endpoints
            such as presigned S3 download URLs.
        **kwargs : Any
            Forwarded to ``httpx.AsyncClient.request`` (e.g. ``params``).

        Returns
        -------
        httpx.Response
            Successful (2xx) response.

        Raises
        ------
        KNMIAuthError
            On HTTP 401/403.
        KNMIHTTPError
            On other HTTP error statuses after exhausting retries.
        """
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.request(
                        method, url, auth=auth, **kwargs
                    )
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise KNMIHTTPError(0, url, f"Transport error: {exc}") from exc
                logger.debug(
                    "Transport error on %s %s (attempt %d): %s",
                    method,
                    url,
                    attempt + 1,
                    exc,
                )
                await asyncio.sleep(await self._backoff_delay(attempt, None))
                continue

            retryable = response.status_code in RETRYABLE_STATUS_CODES
            if not retryable or attempt >= self.max_retries:
                self._raise_for_status(response)
                return response
            logger.debug(
                "Retryable HTTP %d on %s %s (attempt %d)",
                response.status_code,
                method,
                url,
                attempt + 1,
            )
            await asyncio.sleep(await self._backoff_delay(attempt, response))

        raise KNMIHTTPError(0, url, "Request failed without a response.")

    async def get_json(
        self,
        url: str,
        *,
        auth: httpx.Auth | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """GET a URL and return the parsed JSON body.

        Parameters
        ----------
        url : str
            Absolute URL.
        auth : httpx.Auth | None, default=None
            Authentication strategy.
        params : dict | None, default=None
            Query parameters.

        Returns
        -------
        Any
            Parsed JSON payload.
        """
        response = await self.request("GET", url, auth=auth, params=params or {})
        return response.json()

    @asynccontextmanager
    async def stream(
        self, method: str, url: str, *, auth: httpx.Auth | None = None, **kwargs: Any
    ) -> AsyncGenerator[httpx.Response, None]:
        """Stream a response body while holding a concurrency slot.

        Parameters
        ----------
        method : str
            HTTP method (e.g. 'GET').
        url : str
            Absolute URL, typically a presigned S3 download URL.
        auth : httpx.Auth | None, default=None
            Authentication strategy; usually None for presigned URLs.
        **kwargs : Any
            Forwarded to ``httpx.AsyncClient.stream``.

        Yields
        ------
        httpx.Response
            The open response stream.
        """
        stream_ctx = self.client.stream(method, url, auth=auth, **kwargs)
        async with self.semaphore, stream_ctx as response:
            if response.status_code >= 400:
                # error bodies are small; read them so they can be included
                await response.aread()
            self._raise_for_status(response)
            yield response

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.aclose()
