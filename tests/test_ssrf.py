"""SSRF defense and safe network fetching test suite."""

import socket
from unittest.mock import patch

import httpx
import pytest

from edward.services.network import (
    FetchLimitExceededError,
    SSRFSecurityError,
    filter_snapshot_headers,
    is_safe_ip,
    safe_fetch_url,
    validate_url_for_ssrf,
)


def test_is_safe_ip_rejections():
    # Loopback
    assert is_safe_ip("127.0.0.1") is False
    assert is_safe_ip("127.0.0.2") is False
    assert is_safe_ip("::1") is False

    # Private
    assert is_safe_ip("10.0.0.1") is False
    assert is_safe_ip("172.16.0.1") is False
    assert is_safe_ip("192.168.1.1") is False
    assert is_safe_ip("fc00::1") is False

    # Cloud metadata / link-local
    assert is_safe_ip("169.254.169.254") is False
    assert is_safe_ip("fe80::1") is False

    # IPv4-mapped IPv6
    assert is_safe_ip("::ffff:127.0.0.1") is False
    assert is_safe_ip("::ffff:169.254.169.254") is False
    assert is_safe_ip("::ffff:10.0.0.1") is False

    # Unspecified
    assert is_safe_ip("0.0.0.0") is False
    assert is_safe_ip("::") is False

    # Valid public IPs
    assert is_safe_ip("93.184.216.34") is True
    assert is_safe_ip("2606:4700:4700::1111") is True


def test_validate_url_schemes():
    with pytest.raises(SSRFSecurityError, match="insecure scheme"):
        validate_url_for_ssrf("file:///etc/passwd")

    with pytest.raises(SSRFSecurityError, match="insecure scheme"):
        validate_url_for_ssrf("ftp://example.com/file")

    with pytest.raises(SSRFSecurityError, match="Missing or invalid hostname"):
        validate_url_for_ssrf("http://")


def test_validate_url_localhost_and_metadata():
    with pytest.raises(SSRFSecurityError, match="Blocked (loopback|unsafe)"):
        validate_url_for_ssrf("http://localhost/admin")

    with pytest.raises(SSRFSecurityError, match="Blocked (loopback|unsafe)"):
        validate_url_for_ssrf("http://sub.localhost:8080/test")

    with pytest.raises(SSRFSecurityError, match="Blocked (loopback|unsafe)"):
        validate_url_for_ssrf("http://169.254.169.254/latest/meta-data")


@patch("socket.getaddrinfo")
def test_validate_url_dns_resolution_private_ip(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 80))
    ]
    with pytest.raises(SSRFSecurityError, match="resolving to unsafe IP"):
        validate_url_for_ssrf("https://internal.company.corp/api")


@patch("socket.getaddrinfo")
def test_validate_url_dns_resolution_public_ip(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]
    # Should not raise
    validate_url_for_ssrf("https://example.com/article")


def test_filter_snapshot_headers():
    raw_headers = {
        "Content-Type": "text/html; charset=utf-8",
        "Etag": '"123456"',
        "Set-Cookie": "session=secret123",
        "Authorization": "Bearer token456",
        "X-Api-Key": "my-key",
        "Server": "nginx",
        "Content-Length": "1024",
    }
    filtered = filter_snapshot_headers(raw_headers)
    assert filtered["content-type"] == "text/html; charset=utf-8"
    assert filtered["etag"] == '"123456"'
    assert filtered["content-length"] == "1024"
    assert "server" not in filtered
    assert "set-cookie" not in filtered
    assert "authorization" not in filtered
    assert "x-api-key" not in filtered


@patch("edward.services.network.validate_url_for_ssrf")
@patch("httpx.Client.send")
def test_safe_fetch_redirect_to_private_ip_blocked(mock_send, mock_validate):
    # First hop is allowed by initial validation
    mock_validate.side_effect = [
        None,  # First hop pass
        SSRFSecurityError("Host resolves to unsafe IP: 127.0.0.1"),  # Redirect hop blocked
    ]

    # Create a 302 redirect response
    req1 = httpx.Request("GET", "https://example.com/start")
    resp1 = httpx.Response(
        status_code=302,
        headers={"Location": "http://127.0.0.1:8000/internal"},
        request=req1,
    )
    mock_send.return_value = resp1

    with pytest.raises(SSRFSecurityError, match="resolves to unsafe IP"):
        safe_fetch_url("https://example.com/start")


@patch("edward.services.network.validate_url_for_ssrf")
@patch("httpx.Client.send")
def test_safe_fetch_response_too_large(mock_send, mock_validate):
    mock_validate.return_value = None

    req = httpx.Request("GET", "https://example.com/bigfile")
    resp = httpx.Response(
        status_code=200,
        headers={"Content-Length": str(15 * 1024 * 1024)},  # 15MB
        request=req,
    )
    mock_send.return_value = resp

    with pytest.raises(FetchLimitExceededError, match="exceeds limit"):
        safe_fetch_url("https://example.com/bigfile")
