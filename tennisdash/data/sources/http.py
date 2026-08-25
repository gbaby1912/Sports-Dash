"""Shared HTTP fetching: caching, retries, and honest failure reporting.

Every adapter downloads through here so that caching, backoff and - critically -
the distinction between "temporarily unavailable" and "your network blocks this
host" are handled identically everywhere. That distinction matters: retrying a
policy denial wastes minutes and then reports the wrong problem.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ...config import RAW_DIR

log = logging.getLogger(__name__)

USER_AGENT = (
    "Sports-Dash/1.0 (tennis match model; +https://github.com/gbaby1912/Sports-Dash)"
)


class DataUnavailable(RuntimeError):
    """A required remote file could not be retrieved."""


class HostBlocked(DataUnavailable):
    """The network refused the host outright - retrying cannot help.

    Raised on an HTTP 403/407 from a proxy, or on a failed CONNECT tunnel. The
    caller should surface the hostname and the remedy rather than falling back
    to something that looks like data but is not.
    """

    def __init__(self, host: str, detail: str = "") -> None:
        self.host = host
        super().__init__(
            f"Network policy blocks {host}. {detail}\n"
            f"Add {host} to this environment's egress allowlist, or run the "
            f"fetch from a machine with direct internet access."
        )


@dataclass
class FetchReport:
    """What actually happened during a fetch."""

    source: str
    downloaded: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    blocked_hosts: set[str] = field(default_factory=set)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.downloaded or self.cached)

    def summary(self) -> str:
        parts = [f"{len(self.downloaded)} downloaded", f"{len(self.cached)} cached"]
        if self.missing:
            parts.append(f"{len(self.missing)} not published")
        if self.errors:
            parts.append(f"{len(self.errors)} failed")
        if self.blocked_hosts:
            parts.append(f"BLOCKED: {', '.join(sorted(self.blocked_hosts))}")
        return ", ".join(parts)


def cache_path(source: str, filename: str) -> Path:
    directory = RAW_DIR / source
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc or url


def download(
    url: str,
    dest: Path,
    retries: int = 4,
    timeout: int = 60,
    session: requests.Session | None = None,
    min_bytes: int = 64,
) -> Path:
    """GET ``url`` into ``dest``, with backoff on transient failures only."""
    getter = session or requests
    delay = 2.0
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = getter.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.exceptions.ProxyError as exc:
            # A refused CONNECT tunnel is a policy denial wearing a transport
            # error's clothes.
            raise HostBlocked(_host_of(url), str(exc)) from exc
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            continue

        if response.status_code in (403, 407):
            raise HostBlocked(_host_of(url), f"HTTP {response.status_code}")
        if response.status_code == 404:
            raise DataUnavailable(f"not published: {url}")
        if response.status_code >= 500 and attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        if not response.ok:
            raise DataUnavailable(f"HTTP {response.status_code} for {url}")
        if len(response.content) < min_bytes:
            raise DataUnavailable(f"suspiciously small response ({len(response.content)}B): {url}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return dest

    raise DataUnavailable(f"failed after {retries} attempts: {url} ({last_error})")


def download_cached(
    url: str,
    source: str,
    filename: str,
    refresh: bool = False,
    **kwargs,
) -> tuple[Path, bool]:
    """Download unless already cached. Returns (path, was_downloaded)."""
    dest = cache_path(source, filename)
    if dest.exists() and not refresh:
        return dest, False
    try:
        download(url, dest, **kwargs)
        return dest, True
    except DataUnavailable:
        if dest.exists():
            return dest, False  # keep the stale copy rather than losing it
        raise


def check_host(url: str, timeout: int = 15) -> tuple[bool, str]:
    """Probe whether a host is reachable, for the `doctor` command."""
    try:
        response = requests.head(
            url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        if response.status_code in (403, 407):
            return False, f"blocked by network policy (HTTP {response.status_code})"
        if response.status_code >= 400:
            # Some hosts refuse HEAD; a GET of one byte settles it.
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-64"}
            )
            if response.status_code in (403, 407):
                return False, f"blocked by network policy (HTTP {response.status_code})"
            if response.status_code >= 400:
                return False, f"HTTP {response.status_code}"
        return True, "reachable"
    except requests.exceptions.ProxyError as exc:
        return False, f"blocked by network policy (CONNECT refused): {str(exc)[:80]}"
    except requests.exceptions.RequestException as exc:
        return False, f"unreachable: {type(exc).__name__}"
