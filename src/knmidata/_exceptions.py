"""Exception hierarchy for the KNMI Data Platform client."""

from typing import Any


class KNMIError(Exception):
    """Base class for all errors raised by knmidata."""


class KNMIAuthError(KNMIError):
    """Raised when no valid API key can be resolved or the API rejects the key."""


class KNMIHTTPError(KNMIError):
    """Raised when the KNMI API returns an unexpected HTTP status code.

    Parameters
    ----------
    status_code : int
        HTTP status code returned by the API.
    url : str
        Request URL that produced the error.
    body : str
        Raw response body, usually a JSON error message from the gateway.
    """

    def __init__(self, status_code: int, url: str, body: str = "") -> None:
        self.status_code = status_code
        self.url = url
        self.body = body
        message = f"KNMI API request to {url} failed with HTTP {status_code}."
        if body:
            message = f"{message} Response: {body[:500]}"
        super().__init__(message)


class NoFilesFoundError(KNMIError):
    """Raised when a dataset query matches no files."""


def api_error_message(payload: Any) -> str:
    """Extract a human-readable error message from a KNMI JSON error body.

    Parameters
    ----------
    payload : Any
        Parsed JSON body of an error response.

    Returns
    -------
    str
        The extracted message, or an empty string if none is present.
    """
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return ""
