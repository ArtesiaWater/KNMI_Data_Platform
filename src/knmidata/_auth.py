"""Authentication strategies and credential resolution for the KNMI Data Platform."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from logging import getLogger
from typing import Protocol

import httpx

from knmidata._exceptions import KNMIAuthError
from knmidata._typing import AuthFlow
from knmidata.config import ANONYMOUS_KEY_DOCS_URL

logger = getLogger(__name__)

# The anonymous key is published on the docs page as a JWT-like base64url string.
_TOKEN_PATTERN = re.compile(r"eyJ[A-Za-z0-9_-]{50,}")


@lru_cache(maxsize=1)
def get_anonymous_opendata_api_key() -> str:
    """Fetch the anonymous Open Data API key published on the KNMI docs page.

    The key is shared by all anonymous users, rate-limited, and rotated about
    once a year, which is why it is scraped from the documentation page (and
    cached) instead of being hardcoded.

    Returns
    -------
    str
        The anonymous API key.

    Raises
    ------
    KNMIAuthError
        If the key cannot be retrieved (offline, page layout changed, ...).
    """
    try:
        response = httpx.get(
            ANONYMOUS_KEY_DOCS_URL, timeout=30.0, follow_redirects=True
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise KNMIAuthError(
            f"Could not retrieve the anonymous API key from {ANONYMOUS_KEY_DOCS_URL} "
            f"({exc}). Pass `api_key=...` or set the KNMI_API_KEY / KNMI_OPEN_DATA_KEY "
            "environment variable, or register for a key at "
            "https://developer.dataplatform.knmi.nl/."
        ) from exc

    match = _TOKEN_PATTERN.search(response.text)
    if match is None:
        raise KNMIAuthError(
            f"Could not find an anonymous API key on {ANONYMOUS_KEY_DOCS_URL}. "
            "Pass `api_key=...` or set the KNMI_API_KEY / KNMI_OPEN_DATA_KEY "
            "environment variable, or register for a key at "
            "https://developer.dataplatform.knmi.nl/."
        )
    logger.info("Retrieved anonymous Open Data API key from %s", ANONYMOUS_KEY_DOCS_URL)
    return match.group(0)


class Authenticator(Protocol):
    """Protocol defining how credentials attach to HTTP requests."""

    def auth_flow(self, request: httpx.Request) -> AuthFlow:
        """Apply authentication headers to the outgoing request."""
        ...


class NoAuth(httpx.Auth):
    """Strategy for unauthenticated public endpoints."""

    def auth_flow(self, request: httpx.Request) -> AuthFlow:
        """Pass the request through without modifying headers."""
        yield request


@dataclass(slots=True, kw_only=True)
class APIKeyAuth(httpx.Auth):
    """Strategy that injects an API key directly into request headers.

    The KNMI gateway expects the raw key in the ``Authorization`` header
    (no ``Bearer`` prefix).

    Parameters
    ----------
    api_key : str
        The raw API key provided by the KNMI developer platform.
    header_name : str, default="Authorization"
        The header name expected by the gateway.
    """

    api_key: str
    header_name: str = "Authorization"

    def auth_flow(self, request: httpx.Request) -> AuthFlow:
        """Inject the raw API key into the specified header."""
        request.headers[self.header_name] = self.api_key
        yield request


def resolve_open_data_auth(api_key: str | None = None) -> httpx.Auth:
    """Resolve authentication strategy for the Open Data API.

    Resolution order:
    1. Explicitly passed `api_key`.
    2. Environment variable `KNMI_OPEN_DATA_KEY`.
    3. Environment variable `KNMI_API_KEY`.
    4. KNMI public anonymous key (scraped from the docs page, cached).

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
    if not resolved:
        raise KNMIAuthError("Failed to resolve an Open Data API key.")
    return APIKeyAuth(api_key=resolved)


def resolve_edr_auth(api_key: str | None = None) -> httpx.Auth:
    """Resolve authentication strategy for the EDR API.

    The EDR API has no anonymous key; every API key is tied to one API, so an
    Open Data key does not work here.

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
    KNMIAuthError
        If no key can be resolved from arguments or environment variables.
    """
    resolved = api_key or os.getenv("KNMI_EDR_KEY") or os.getenv("KNMI_API_KEY")
    if not resolved:
        raise KNMIAuthError(
            "The KNMI EDR API requires a personal registered API key. "
            "Pass `edr_key=...` or set the KNMI_EDR_KEY environment variable. "
            "Request a key at https://developer.dataplatform.knmi.nl/."
        )
    return APIKeyAuth(api_key=resolved)
