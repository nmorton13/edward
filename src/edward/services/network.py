"""Safe network client with comprehensive SSRF protection and response size limits."""

import hashlib
import ipaddress
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_REDIRECTS = 5

# Strict header allowlist for source snapshots
SNAPSHOT_HEADER_ALLOWLIST = {
    "content-type",
    "content-length",
    "etag",
    "last-modified",
    "date",
}


class NetworkError(Exception):
    """Base error for network operations."""

    pass


class SSRFError(NetworkError):
    """Raised when a URL targets a private, loopback, or metadata address."""

    pass


SSRFSecurityError = SSRFError


class ResponseTooLargeError(NetworkError):
    """Raised when a response exceeds the maximum allowed payload size."""

    pass


FetchLimitExceededError = ResponseTooLargeError


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    content_hash: str
    elapsed_seconds: float


FetchedResource = FetchResult


def is_safe_ip(ip_str: str) -> bool:
    """Validate that an IP address is a public, non-private, non-loopback, non-metadata address."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    # Handle IPv6-mapped IPv4 addresses (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return is_safe_ip(str(ip.ipv4_mapped))

    # Reject private, loopback, link-local, multicast, reserved, unspecified
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    # Cloud metadata specific checks (e.g. 169.254.169.254, fd00:ec2::254)
    if isinstance(ip, ipaddress.IPv4Address):
        # 169.254.0.0/16 is covered by is_link_local, but check explicitly
        if ip in ipaddress.ip_network("169.254.0.0/16"):
            return False
        # 0.0.0.0/8 (current network)
        if ip in ipaddress.ip_network("0.0.0.0/8"):
            return False
        # 100.64.0.0/10 (Carrier-grade NAT)
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return False
        # 192.0.0.0/24 (IETF Protocol Assignments)
        if ip in ipaddress.ip_network("192.0.0.0/24"):
            return False

    elif isinstance(ip, ipaddress.IPv6Address):
        # Unique local (fc00::/7)
        if ip in ipaddress.ip_network("fc00::/7"):
            return False
        # Link-local (fe80::/10)
        if ip in ipaddress.ip_network("fe80::/10"):
            return False

    return True


def validate_url_for_ssrf(url: str) -> None:
    """Check a URL against SSRF vulnerabilities by validating scheme, hostname, and all resolved IPs."""
    clean = url.strip()
    if not clean:
        raise SSRFError("URL cannot be empty")

    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise SSRFError(
            f"Blocked unsupported or insecure scheme '{scheme}': only http and https are allowed"
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError(f"Missing or invalid hostname in URL: '{url}'")

    hostname_clean = hostname.strip().lower()
    if hostname_clean in ("localhost", "local", "broadcasthost") or hostname_clean.endswith(
        ".localhost"
    ):
        raise SSRFError(f"Blocked loopback hostname: '{hostname}'")

    # Check if hostname is an IP literal
    try:
        ipaddress.ip_address(hostname_clean)
        is_ip = True
    except ValueError:
        is_ip = False

    if is_ip:
        if not is_safe_ip(hostname_clean):
            raise SSRFError(f"Blocked unsafe IP literal: '{hostname_clean}'")
        return

    # Resolve hostname via DNS and check all returned IPs
    try:
        addr_info = socket.getaddrinfo(hostname_clean, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFError(f"Failed to resolve hostname '{hostname_clean}': {e}") from e

    if not addr_info:
        raise SSRFError(f"No IP addresses resolved for hostname '{hostname_clean}'")

    for item in addr_info:
        ip_addr = item[4][0]
        if not is_safe_ip(ip_addr):
            raise SSRFError(
                f"Blocked hostname '{hostname_clean}' resolving to unsafe IP '{ip_addr}'"
            )


class SSRFSafeSyncBackend(httpcore.SyncBackend):
    """Sync backend that verifies the connected socket peer IP before any data is exchanged, defeating DNS rebinding."""

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        stream = super().connect_tcp(
            host=host,
            port=port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        try:
            sock = getattr(stream, "_sock", None)
            if sock is not None:
                peer_ip = sock.getpeername()[0]
                if not is_safe_ip(peer_ip):
                    sock.close()
                    raise SSRFError(
                        f"Blocked connection to unsafe IP '{peer_ip}' for host '{host}'"
                    )
        except SSRFError:
            raise
        except Exception as e:
            raise SSRFError(f"Failed to verify socket peer address for host '{host}': {e}") from e
        return stream


class SSRFSafeHTTPTransport(httpx.HTTPTransport):
    """HTTP transport configured with SSRFSafeSyncBackend to prevent DNS rebinding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool = httpcore.ConnectionPool(
            network_backend=SSRFSafeSyncBackend(),
            retries=0,
        )


def filter_snapshot_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter response headers against the snapshot allowlist, stripping all auth/cookie/proxy data."""
    return {k.lower(): str(v) for k, v in headers.items() if k.lower() in SNAPSHOT_HEADER_ALLOWLIST}


def safe_fetch_url(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_redirects: int = MAX_REDIRECTS,
    user_agent: str = "Edward/0.1.0 (Personal Research Memory)",
) -> FetchResult:
    """Fetch content from a remote URL safely with SSRF protection, size caps, and redirect validation."""
    current_url = url
    redirect_count = 0
    start_time = time.monotonic()

    custom_headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    }

    transport = SSRFSafeHTTPTransport(verify=True)
    with httpx.Client(
        transport=transport,
        follow_redirects=False,
        timeout=httpx.Timeout(timeout, connect=5.0),
    ) as client:
        while True:
            # Revalidate SSRF on every hop
            validate_url_for_ssrf(current_url)

            try:
                with client.stream("GET", current_url, headers=custom_headers) as response:
                    # Handle redirects explicitly to recheck SSRF
                    if response.is_redirect:
                        redirect_count += 1
                        if redirect_count > max_redirects:
                            raise NetworkError(
                                f"Exceeded maximum redirect limit of {max_redirects}"
                            )

                        location = response.headers.get("location")
                        if not location:
                            raise NetworkError(
                                f"Redirect status {response.status_code} without Location header"
                            )

                        current_url = urljoin(current_url, location)
                        continue

                    # Check declared Content-Length
                    cl_header = response.headers.get("content-length")
                    if cl_header and cl_header.isdigit() and int(cl_header) > max_bytes:
                        raise ResponseTooLargeError(
                            f"Response size {cl_header} bytes exceeds limit of {max_bytes} bytes"
                        )

                    # Stream content with incremental chunk cap
                    chunks: list[bytes] = []
                    received_bytes = 0
                    for chunk in response.iter_bytes(chunk_size=65536):
                        received_bytes += len(chunk)
                        if received_bytes > max_bytes:
                            raise ResponseTooLargeError(
                                f"Response content length {received_bytes} exceeds limit of {max_bytes} bytes"
                            )
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    elapsed = time.monotonic() - start_time
                    content_hash = hashlib.sha256(body).hexdigest()
                    filtered_headers = filter_snapshot_headers(dict(response.headers))

                    return FetchResult(
                        url=url,
                        final_url=current_url,
                        status_code=response.status_code,
                        headers=filtered_headers,
                        body=body,
                        content_hash=content_hash,
                        elapsed_seconds=elapsed,
                    )
            except (SSRFError, ResponseTooLargeError, NetworkError):
                raise
            except httpx.RequestError as e:
                if e.__cause__ and isinstance(e.__cause__, SSRFError):
                    raise e.__cause__ from e
                raise NetworkError(f"HTTP request failed for '{current_url}': {e}") from e
