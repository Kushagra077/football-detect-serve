"""app/main.py's SSRF guard for image_url. No network calls - scheme checks are
pure, and hostname resolution only ever uses "localhost" (guaranteed to resolve
via loopback with no real network access, even in CI)."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import time

import pytest

import app.main as main_module
from app.main import _is_blocked_ip, _validate_image_url


@pytest.mark.parametrize(
    "ip_str",
    [
        "127.0.0.1",        # loopback
        "169.254.169.254",  # cloud metadata endpoint (link-local)
        "10.0.0.1",         # private
        "172.16.0.1",       # private
        "192.168.1.1",      # private
        "0.0.0.0",          # unspecified
        "224.0.0.1",        # multicast
        "::1",              # loopback, IPv6
    ],
)
def test_is_blocked_ip_blocks_internal_addresses(ip_str):
    assert _is_blocked_ip(ipaddress.ip_address(ip_str)) is True


@pytest.mark.parametrize("ip_str", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
def test_is_blocked_ip_allows_public_addresses(ip_str):
    assert _is_blocked_ip(ipaddress.ip_address(ip_str)) is False


def test_validate_image_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        _validate_image_url("file:///etc/passwd")


def test_validate_image_url_rejects_ftp_scheme():
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        _validate_image_url("ftp://example.com/a.jpg")


def test_validate_image_url_rejects_missing_hostname():
    with pytest.raises(ValueError, match="no hostname"):
        _validate_image_url("http:///path-with-no-host")


def test_validate_image_url_rejects_localhost():
    with pytest.raises(ValueError, match="private/internal address"):
        _validate_image_url("http://localhost/image.jpg")


def test_validate_image_url_rejects_loopback_literal():
    with pytest.raises(ValueError, match="private/internal address"):
        _validate_image_url("http://127.0.0.1/image.jpg")


def test_validate_image_url_rejects_unresolvable_host():
    with pytest.raises(ValueError, match="could not resolve host"):
        _validate_image_url("http://this-host-does-not-exist.invalid/x.jpg")


async def test_fetch_url_dns_lookup_does_not_block_event_loop(monkeypatch):
    """Guards the fixed bug: _validate_image_url's DNS lookup used to run
    directly on the event loop, before _fetch_url dispatched to the executor -
    a slow resolver would then stall every other request on the worker. It
    must now run inside the same executor call as the fetch itself, so a slow
    lookup only delays this one request, never the event loop.

    No real network or DNS: socket.getaddrinfo and requests.get are both
    replaced with fakes; getaddrinfo's fake blocks for 200ms to simulate a
    slow resolver.
    """

    def slow_getaddrinfo(host, *args, **kwargs):
        time.sleep(0.2)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    class _FakeResponse:
        def raise_for_status(self):
            pass

        class raw:
            @staticmethod
            def read(*args, **kwargs):
                return b"fake-image-bytes"

    monkeypatch.setattr(main_module.socket, "getaddrinfo", slow_getaddrinfo)
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse())

    first_tick_time: list[float] = []

    async def ticker() -> None:
        for _ in range(5):
            first_tick_time.append(time.perf_counter())
            await asyncio.sleep(0.01)

    t0 = time.perf_counter()
    # Scheduled, not run yet - a Task only starts once its creator yields
    # control back to the loop, which is exactly what this test is probing.
    ticker_task = asyncio.ensure_future(ticker())
    data = await main_module._fetch_url("http://example.com/x.jpg")
    fetch_elapsed = time.perf_counter() - t0
    await ticker_task

    assert data == b"fake-image-bytes"
    assert fetch_elapsed >= 0.2  # the slow "DNS lookup" really did happen
    # The ticker must get its first turn almost immediately: _fetch_url's
    # first await point should be run_in_executor (yields right away, work
    # happens on a background thread). If the DNS lookup instead ran
    # synchronously on the event loop before any await, the ticker couldn't
    # start until that ~200ms finished, and this would be ~0.2 instead.
    assert first_tick_time[0] - t0 < 0.05
