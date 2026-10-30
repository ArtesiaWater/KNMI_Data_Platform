"""Tests for the HTTP transport: retries, error mapping, streaming, semaphore."""

import asyncio

import httpx
import pytest
import respx

from knmidata._exceptions import KNMIAuthError, KNMIHTTPError

URL = "https://api.dataplatform.knmi.nl/open-data/v1/ping"


async def test_request_returns_response(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
        response = await transport.request("GET", URL)
        assert response.status_code == 200
        assert response.json() == {"ok": True}


async def test_request_retries_on_500_then_succeeds(transport):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(URL).mock(
            side_effect=[
                httpx.Response(500, text="server error"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        response = await transport.request("GET", URL)
        assert response.status_code == 200
        assert route.call_count == 2


async def test_request_honors_retry_after(transport):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        response = await transport.request("GET", URL)
        assert response.status_code == 200
        assert route.call_count == 2


async def test_request_gives_up_after_max_retries(transport):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(URL).mock(return_value=httpx.Response(503, text="down"))
        with pytest.raises(KNMIHTTPError) as exc_info:
            await transport.request("GET", URL)
        assert exc_info.value.status_code == 503
        assert route.call_count == transport.max_retries + 1


async def test_request_retries_on_transport_error(transport):
    with respx.mock(assert_all_called=False) as router:
        route = router.get(URL).mock(
            side_effect=[
                httpx.ConnectError("boom"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        response = await transport.request("GET", URL)
        assert response.status_code == 200
        assert route.call_count == 2


async def test_request_transport_error_exhausted(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(KNMIHTTPError, match="Transport error"):
            await transport.request("GET", URL)


async def test_403_maps_to_auth_error(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(
            return_value=httpx.Response(
                403, json={"error": "Authorization field missing"}
            )
        )
        with pytest.raises(KNMIAuthError, match="403"):
            await transport.request("GET", URL)


async def test_400_maps_to_http_error_with_body(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(return_value=httpx.Response(400, text="bad request body"))
        with pytest.raises(KNMIHTTPError) as exc_info:
            await transport.request("GET", URL)
        assert exc_info.value.status_code == 400
        assert "bad request body" in str(exc_info.value)


async def test_error_message_extracted_from_json_body(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(
            return_value=httpx.Response(400, json={"error": "dataset not found"})
        )
        with pytest.raises(KNMIHTTPError, match="dataset not found"):
            await transport.request("GET", URL)


async def test_stream_yields_body(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(return_value=httpx.Response(200, content=b"x" * 1000))
        chunks = []
        async with transport.stream("GET", URL) as response:
            async for chunk in response.aiter_bytes(chunk_size=128):
                chunks.append(chunk)
        assert b"".join(chunks) == b"x" * 1000


async def test_stream_maps_errors(transport):
    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(return_value=httpx.Response(404, text="nope"))
        with pytest.raises(KNMIHTTPError) as exc_info:
            async with transport.stream("GET", URL):
                pass
        assert exc_info.value.status_code == 404


async def test_error_messages_strip_presigned_query_string(transport):
    # the query string of a presigned URL is a credential: it must never end
    # up in an exception message or the stored url attribute
    url = "https://fake-s3.amazonaws.com/f.nc?X-Amz-Signature=topsecret&X-Amz-Credential=abc"
    with respx.mock(assert_all_called=False) as router:
        router.get(url).mock(return_value=httpx.Response(404, text="nope"))
        with pytest.raises(KNMIHTTPError) as exc_info:
            await transport.request("GET", url)
        assert exc_info.value.status_code == 404
        assert "topsecret" not in str(exc_info.value)
        assert exc_info.value.url == "https://fake-s3.amazonaws.com/f.nc"
    with respx.mock(assert_all_called=False) as router:
        router.get(url).mock(return_value=httpx.Response(403, text="forbidden"))
        with pytest.raises(KNMIAuthError) as exc_info:
            await transport.request("GET", url)
        assert "topsecret" not in str(exc_info.value)


async def test_stream_error_strips_presigned_signature(transport):
    url = "https://fake-s3.amazonaws.com/f.nc?X-Amz-Signature=topsecret"
    with respx.mock(assert_all_called=False) as router:
        router.get(url).mock(return_value=httpx.Response(403, text="denied"))
        with pytest.raises(KNMIAuthError) as exc_info:
            async with transport.stream("GET", url):
                pass
    assert "topsecret" not in str(exc_info.value)


def test_knmidata_logger_nullhandler_and_debug_level():
    # library pattern: only a NullHandler, but our own (sanitized) log lines
    # default to DEBUG so basicConfig(level=DEBUG) is enough to see them
    import logging

    import knmidata  # noqa: F401

    logger = logging.getLogger("knmidata")
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_httpx_logger_left_untouched():
    # knmidata must not configure third-party loggers; users decide themselves
    # (httpx logs full URLs at INFO, so see the README caution about DEBUG)
    import logging

    import knmidata  # noqa: F401

    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level == logging.NOTSET
    assert not httpx_logger.filters
    assert not httpx_logger.handlers


async def test_semaphore_bounds_concurrency(transport):
    transport.semaphore = asyncio.Semaphore(2)
    active = 0
    peak = 0

    async def slow(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return httpx.Response(200, json={"ok": True})

    with respx.mock(assert_all_called=False) as router:
        router.get(URL).mock(side_effect=slow)
        await asyncio.gather(*[transport.request("GET", URL) for _ in range(6)])
        assert peak <= 2
