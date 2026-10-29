"""HTTP transport layer with concurrency control and connection pooling."""

import asyncio
from dataclasses import dataclass, field

import httpx


@dataclass(slots=True, kw_only=True)
class AsyncTransport:
    """Encapsulates an httpx.AsyncClient session bounded by a concurrency semaphore.

    Parameters
    ----------
    max_concurrency : int, default=15
        Maximum number of concurrent network requests permitted across the session.
    timeout : float, default=60.0
        Read/connect timeout in seconds.
    """

    max_concurrency: int = 15
    timeout: float = 60.0
    client: httpx.AsyncClient = field(init=False)
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the semaphore and underlying httpx client."""
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=self.max_concurrency * 2,
                max_keepalive_connections=self.max_concurrency,
            ),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()
