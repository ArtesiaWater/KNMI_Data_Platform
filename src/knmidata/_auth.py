"""Authentication strategies and credential resolution for the KNMI Data Platform."""

import os
from collections.abc import Generator
from dataclasses import dataclass
from logging import getLogger
from typing import Protocol

import httpx
from pandas import Timestamp

logger = getLogger(__name__)


def get_anonymous_opendata_api_key() -> str | None:
    """Get anonymous API Key from KNMI data platform."""
    url = "https://developer.dataplatform.knmi.nl/open-data-api#token"
    try:
        webpage = httpx.get(url, timeout=120)  # get webpage
        webpage.raise_for_status()  # raise error if not 200
        api_key = (
            webpage.text.split("</code></pre>")[0].split("<pre><code>")[-1].strip()
        )  # obtain apikey from codeblock on webpage
        if len(api_key) != 120:
            msg = (
                f"Could not obtain API Key from {url}, trying API "
                f"Key from memory. Found API Key = {api_key}"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Retrieved anonymous API Key from {url}")
        return api_key
    except httpx.HTTPError as e:
        api_key_memory_date = "2027-07-01"
        if Timestamp.today() < Timestamp(api_key_memory_date):
            logger.info(
                "Retrieved anonymous API Key (available till"
                f" {api_key_memory_date}) from memory"
            )
            api_key = (
                "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1"
                "ZDBhMmQ5YzRkYzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9"
            )
            return api_key
        else:
            raise httpx.HTTPError(
                f"Could not retrieve anonymous API Key from {url}. {e} Please"
                " create your own API key at https://developer.dataplatform.knmi.nl/"
            )


class Authenticator(Protocol):
    """Protocol defining how credentials attach to HTTP requests."""

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Apply authentication headers to the outgoing request."""
        ...


class NoAuth(httpx.Auth):
    """Strategy for unauthenticated public endpoints."""

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Pass the request through without modifying headers."""
        yield request


@dataclass(slots=True, kw_only=True)
class APIKeyAuth(httpx.Auth):
    """Strategy that injects an API key directly into request headers.

    Parameters
    ----------
    api_key : str
        The raw API key provided by the KNMI developer platform.
    header_name : str, default="Authorization"
        The header name expected by the gateway.
    """

    api_key: str
    header_name: str = "Authorization"

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Inject the raw API key into the specified header without Bearer prefix."""
        request.headers[self.header_name] = self.api_key
        yield request


def resolve_open_data_auth(api_key: str | None = None) -> httpx.Auth:
    """Resolve authentication strategy for the Open Data API.

    Resolution order:
    1. Explicitly passed `api_key`.
    2. Environment variable `KNMI_OPEN_DATA_KEY`.
    3. Environment variable `KNMI_API_KEY`.
    4. KNMI public anonymous key fallback.

    Parameters
    ----------
    api_key : str | None, default=None
        Optional explicit API key.

    Returns
    -------
    httpx.Auth
        Configured authentication strategy.
    """
    resolved = (
        api_key
        or os.getenv("KNMI_OPEN_DATA_KEY")
        or os.getenv("KNMI_API_KEY")
        or get_anonymous_opendata_api_key()
    )
    assert resolved is not None, "Failed to resolve an Open Data API key."
    return APIKeyAuth(api_key=resolved)


def resolve_edr_auth(api_key: str | None = None) -> httpx.Auth:
    """Resolve authentication strategy for the EDR API.

    Resolution order:
    1. Explicitly passed `api_key`.
    2. Environment variable `KNMI_EDR_KEY`.
    3. Environment variable `KNMI_API_KEY`.

    Parameters
    ----------
    api_key : str | None, default=None
        Optional explicit API key.

    Returns
    -------
    httpx.Auth
        Configured authentication strategy.

    Raises
    ------
    ValueError
        If no key can be resolved from arguments or environment variables.
    """
    resolved = api_key or os.getenv("KNMI_EDR_KEY") or os.getenv("KNMI_API_KEY")
    if not resolved:
        raise ValueError(
            "KNMI EDR API requires a personal registered API key. "
            "Pass `edr_key=...` or set the `KNMI_EDR_KEY` environment variable."
        )
    return APIKeyAuth(api_key=resolved)
