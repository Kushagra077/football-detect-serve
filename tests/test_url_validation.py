"""app/main.py's SSRF guard for image_url. No network calls - scheme checks are
pure, and hostname resolution only ever uses "localhost" (guaranteed to resolve
via loopback with no real network access, even in CI)."""
from __future__ import annotations

import ipaddress

import pytest

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
