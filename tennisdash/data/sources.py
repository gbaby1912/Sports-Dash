"""Download and cache the canonical public match archives.

Primary source is Jeff Sackmann's match archives, which are the de-facto
standard for tennis modelling because they carry per-match serve counting
stats (aces, double faults, service points, first-serve in/won, break points
faced/saved) rather than just results. Those counting stats are what make
serve/return skill estimation possible at all.

Everything is cached on disk under ``data/raw`` so a rebuild is cheap and the
pipeline works offline once seeded.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from ..config import RAW_DIR

log = logging.getLogger(__name__)

SACKMANN_REPOS = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}

# Odds archives, used only to benchmark the model against the closing market.
TENNIS_DATA_BASE = "http://www.tennis-data.co.uk"

DEFAULT_START_YEAR = 2000
_USER_AGENT = "Sports-Dash/1.0 (tennis match model; +https://github.com/gbaby1912/Sports-Dash)"


class DataUnavailable(RuntimeError):
    """Raised when a required remote archive cannot be reached."""


@dataclass
class FetchReport:
    """What actually happened during a fetch, for honest logging."""

    downloaded: list[str]
    cached: list[str]
    failed: list[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return bool(self.downloaded or self.cached)

    def summary(self) -> str:
        return (
            f"{len(self.downloaded)} downloaded, {len(self.cached)} from cache, "
            f"{len(self.failed)} failed"
        )


def _cache_path(tour: str, filename: str) -> Path:
    directory = RAW_DIR / tour
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _download(url: str, dest: Path, retries: int = 4, timeout: int = 60) -> None:
    """GET ``url`` into ``dest`` with exponential backoff on transient errors."""
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
            if response.status_code == 404:
                raise DataUnavailable(f"404 Not Found: {url}")
            if response.status_code in (403, 407):
                # Egress policy denial - retrying will not help and the caller
                # needs to hear about it rather than silently getting nothing.
                raise DataUnavailable(
                    f"HTTP {response.status_code} for {url}. The host is blocked by "
                    "network policy; run the ingest from a machine with access."
                )
            response.raise_for_status()
            dest.write_bytes(response.content)
            return
        except DataUnavailable:
            raise
        except Exception as exc:  # network flakiness
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    raise DataUnavailable(f"Failed to download {url}: {last_error}")


def fetch_matches(
    tour: str,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    refresh_last_n_years: int = 1,
) -> FetchReport:
    """Download season match files for one tour.

    Historical seasons never change, so a cached file is reused unconditionally.
    The most recent ``refresh_last_n_years`` seasons are always re-fetched
    because they are appended to during the year.
    """
    tour = tour.lower()
    if tour not in SACKMANN_REPOS:
        raise ValueError(f"unknown tour {tour!r}")
    end_year = end_year or pd.Timestamp.today().year

    downloaded: list[str] = []
    cached: list[str] = []
    failed: list[tuple[str, str]] = []

    for year in range(start_year, end_year + 1):
        filename = f"{tour}_matches_{year}.csv"
        dest = _cache_path(tour, filename)
        is_recent = year > end_year - refresh_last_n_years
        if dest.exists() and not is_recent:
            cached.append(filename)
            continue
        url = f"{SACKMANN_REPOS[tour]}/{filename}"
        try:
            _download(url, dest)
            downloaded.append(filename)
        except DataUnavailable as exc:
            if dest.exists():
                cached.append(filename)
            else:
                failed.append((filename, str(exc)))

    return FetchReport(downloaded, cached, failed)


def fetch_players(tour: str) -> Path | None:
    """Download the player reference table (DOB, hand, height, country)."""
    tour = tour.lower()
    filename = f"{tour}_players.csv"
    dest = _cache_path(tour, filename)
    if dest.exists():
        return dest
    try:
        _download(f"{SACKMANN_REPOS[tour]}/{filename}", dest)
        return dest
    except DataUnavailable as exc:
        log.warning("player table unavailable for %s: %s", tour, exc)
        return None


def fetch_rankings(tour: str, decades: tuple[str, ...] = ("00s", "10s", "20s")) -> list[Path]:
    """Download historical ranking snapshots, used for rank-trend features."""
    tour = tour.lower()
    paths: list[Path] = []
    for decade in decades:
        filename = f"{tour}_rankings_{decade}.csv"
        dest = _cache_path(tour, filename)
        if dest.exists():
            paths.append(dest)
            continue
        try:
            _download(f"{SACKMANN_REPOS[tour]}/{filename}", dest)
            paths.append(dest)
        except DataUnavailable as exc:
            log.warning("rankings unavailable (%s): %s", filename, exc)
    # The current-season file is named differently.
    current = _cache_path(tour, f"{tour}_rankings_current.csv")
    if not current.exists():
        try:
            _download(f"{SACKMANN_REPOS[tour]}/{tour}_rankings_current.csv", current)
        except DataUnavailable:
            pass
    if current.exists():
        paths.append(current)
    return paths


def fetch_odds(tour: str, start_year: int = 2010, end_year: int | None = None) -> list[Path]:
    """Download closing-odds archives (optional, benchmarking only).

    The model never trains on odds - that would be circular. They exist purely
    so the backtest can answer "is this model better than the market?", which
    is the only honest external yardstick for a tennis model.
    """
    end_year = end_year or pd.Timestamp.today().year
    suffix = "" if tour.lower() == "atp" else "w"
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        dest = _cache_path(f"{tour}_odds", f"{year}.xlsx")
        if dest.exists():
            paths.append(dest)
            continue
        url = f"{TENNIS_DATA_BASE}/{year}{suffix}/{year}.xlsx"
        try:
            _download(url, dest, retries=2, timeout=45)
            paths.append(dest)
        except DataUnavailable as exc:
            log.info("odds unavailable for %s %s: %s", tour, year, exc)
    return paths


def load_cached_matches(tour: str) -> pd.DataFrame:
    """Concatenate every cached season file for a tour."""
    directory = RAW_DIR / tour.lower()
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.glob(f"{tour.lower()}_matches_*.csv")):
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except Exception as exc:  # corrupt partial download
            log.warning("skipping unreadable %s: %s", path.name, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), low_memory=False)
