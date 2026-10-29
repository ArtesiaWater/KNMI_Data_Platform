"""Tests for authentication strategies and credential resolution."""

import httpx
import pytest
import respx

from knmidata._auth import (
    APIKeyAuth,
    NoAuth,
    get_anonymous_opendata_api_key,
    resolve_edr_auth,
    resolve_open_data_auth,
)
from knmidata._constants import ANONYMOUS_KEY_DOCS_URL
from knmidata._exceptions import KNMIAuthError

FAKE_KEY = "eyJ" + "aB3" * 40  # JWT-like token, 122 chars


def test_api_key_auth_sets_raw_authorization_header():
    auth = APIKeyAuth(api_key="secret-key")
    request = httpx.Request("GET", "https://example.com")
    flow = auth.auth_flow(request)
    outgoing = next(flow)
    assert outgoing.headers["Authorization"] == "secret-key"


def test_no_auth_passes_request_through():
    request = httpx.Request("GET", "https://example.com")
    flow = NoAuth().auth_flow(request)
    outgoing = next(flow)
    assert "Authorization" not in outgoing.headers


def test_resolve_open_data_auth_prefers_explicit_key(monkeypatch):
    monkeypatch.setenv("KNMI_API_KEY", "env-key")
    monkeypatch.setenv("KNMI_OPEN_DATA_KEY", "open-data-env-key")
    auth = resolve_open_data_auth("explicit-key")
    assert isinstance(auth, APIKeyAuth)
    assert auth.api_key == "explicit-key"


def test_resolve_open_data_auth_env_order(monkeypatch):
    monkeypatch.setenv("KNMI_API_KEY", "env-key")
    assert resolve_open_data_auth().api_key == "env-key"
    monkeypatch.setenv("KNMI_OPEN_DATA_KEY", "open-data-env-key")
    assert resolve_open_data_auth().api_key == "open-data-env-key"


def test_resolve_edr_auth_requires_a_key():
    with pytest.raises(KNMIAuthError, match="EDR"):
        resolve_edr_auth()


def test_resolve_edr_auth_env(monkeypatch):
    monkeypatch.setenv("KNMI_EDR_KEY", "edr-env-key")
    assert resolve_edr_auth().api_key == "edr-env-key"
    assert resolve_edr_auth("explicit").api_key == "explicit"


def test_anonymous_key_scraped_and_cached():
    html = (
        "<html><body><p>docs</p>"
        f"<pre><code>{FAKE_KEY}</code></pre>"
        "<pre><code>curl ...</code></pre>"
        "</body></html>"
    )
    with respx.mock(assert_all_called=False) as router:
        route = router.get(ANONYMOUS_KEY_DOCS_URL).mock(
            return_value=httpx.Response(200, text=html)
        )
        assert get_anonymous_opendata_api_key() == FAKE_KEY
        # second call must be served from the cache, not the network
        assert get_anonymous_opendata_api_key() == FAKE_KEY
        assert route.call_count == 1


def test_anonymous_key_failure_raises_auth_error():
    with respx.mock(assert_all_called=False) as router:
        router.get(ANONYMOUS_KEY_DOCS_URL).mock(
            side_effect=httpx.ConnectError("offline")
        )
        with pytest.raises(KNMIAuthError, match="anonymous API key"):
            get_anonymous_opendata_api_key()


def test_anonymous_key_not_found_on_page():
    with respx.mock(assert_all_called=False) as router:
        router.get(ANONYMOUS_KEY_DOCS_URL).mock(
            return_value=httpx.Response(200, text="<html>no key here</html>")
        )
        with pytest.raises(KNMIAuthError, match="anonymous API key"):
            get_anonymous_opendata_api_key()


def test_resolve_open_data_auth_falls_back_to_anonymous():
    html = f"<pre><code>{FAKE_KEY}</code></pre>"
    with respx.mock(assert_all_called=False) as router:
        router.get(ANONYMOUS_KEY_DOCS_URL).mock(
            return_value=httpx.Response(200, text=html)
        )
        auth = resolve_open_data_auth()
        assert auth.api_key == FAKE_KEY
