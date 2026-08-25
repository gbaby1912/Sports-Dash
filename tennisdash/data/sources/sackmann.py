"""Jeff Sackmann's archives - Tennis Abstract's underlying data.

Three repositories, all maintained by Jeff Sackmann and all published openly.
Together they are the most complete public record of professional tennis:

``tennis_atp`` / ``tennis_wta``
    One CSV per season since 1968, with results *and per-match serve counting
    stats*: aces, double faults, service points, first serves in and won,
    second-serve points won, service games, break points faced and saved. Those
    counting stats are the reason this is the modelling backbone - without them
    there is no serve/return skill estimation, and the model collapses back to
    an Elo-plus-form system.

``tennis_MatchChartingProject``
    Shot-by-shot charting of thousands of matches: serve direction (wide, body,
    down the T), return depth, rally length, shot types, net points, and
    behaviour on key points. This is the deepest tactical data available
    publicly, and it is what lets the model distinguish *how* a player wins
    points rather than only how many.

The charting data covers a few thousand matches rather than every match, so it
is used as an enrichment layer - a player-level style profile that improves the
matchups it covers - never as a required input.
"""
from __future__ import annotations

import logging

import pandas as pd

from ...config import RAW_DIR
from .http import DataUnavailable, FetchReport, HostBlocked, cache_path, download_cached

log = logging.getLogger(__name__)

HOST = "raw.githubusercontent.com"
RAW_BASE = "https://raw.githubusercontent.com/JeffSackmann"

REPOS = {
    "atp": "tennis_atp",
    "wta": "tennis_wta",
    "charting": "tennis_MatchChartingProject",
}

SOURCE = "sackmann"
FIRST_SEASON = 1968
DEFAULT_START_YEAR = 2000

# Match Charting Project tables, keyed by the suffix used in the filename.
# "m" and "w" are the men's and women's files.
CHARTING_TABLES = [
    "matches",
    "stats-Overview",
    "stats-ServeBasics",
    "stats-ServeDirection",
    "stats-ServeInfluence",
    "stats-ReturnOutcomes",
    "stats-ReturnDepth",
    "stats-RallyOutcomes",
    "stats-ShotTypes",
    "stats-ShotDirection",
    "stats-NetPoints",
    "stats-KeyPointsServe",
    "stats-KeyPointsReturn",
    "stats-SnV",
]


def matches_url(tour: str, year: int) -> str:
    return f"{RAW_BASE}/{REPOS[tour]}/master/{tour}_matches_{year}.csv"


def charting_url(table: str, gender: str = "m") -> str:
    return f"{RAW_BASE}/{REPOS['charting']}/master/charting-{gender}-{table}.csv"


def fetch_matches(
    tours: tuple[str, ...] = ("atp", "wta"),
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    refresh_current: bool = True,
) -> FetchReport:
    """Download season match files. Completed seasons are cached permanently."""
    end_year = end_year or pd.Timestamp.today().year
    report = FetchReport(source=SOURCE)

    for tour in tours:
        if tour not in REPOS:
            raise ValueError(f"unknown tour {tour!r}")
        for year in range(start_year, end_year + 1):
            filename = f"{tour}_matches_{year}.csv"
            # Past seasons never change; the current one is appended to weekly.
            refresh = refresh_current and year >= end_year
            try:
                _, downloaded = download_cached(
                    matches_url(tour, year), f"{SOURCE}/{tour}", filename, refresh=refresh
                )
                (report.downloaded if downloaded else report.cached).append(filename)
            except HostBlocked as exc:
                report.blocked_hosts.add(exc.host)
                return report
            except DataUnavailable as exc:
                report.missing.append(filename)
                log.debug("%s: %s", filename, exc)

    return report


def fetch_reference(tours: tuple[str, ...] = ("atp", "wta")) -> FetchReport:
    """Player biographies and historical ranking snapshots."""
    report = FetchReport(source=f"{SOURCE}-reference")
    for tour in tours:
        targets = [f"{tour}_players.csv"] + [
            f"{tour}_rankings_{decade}.csv" for decade in ("00s", "10s", "20s", "current")
        ]
        for filename in targets:
            url = f"{RAW_BASE}/{REPOS[tour]}/master/{filename}"
            refresh = filename.endswith("current.csv")
            try:
                _, downloaded = download_cached(
                    url, f"{SOURCE}/{tour}", filename, refresh=refresh
                )
                (report.downloaded if downloaded else report.cached).append(filename)
            except HostBlocked as exc:
                report.blocked_hosts.add(exc.host)
                return report
            except DataUnavailable:
                report.missing.append(filename)
    return report


def fetch_charting(genders: tuple[str, ...] = ("m", "w")) -> FetchReport:
    """Download the Match Charting Project stat tables."""
    report = FetchReport(source=f"{SOURCE}-charting")
    for gender in genders:
        for table in CHARTING_TABLES:
            filename = f"charting-{gender}-{table}.csv"
            try:
                _, downloaded = download_cached(
                    charting_url(table, gender), f"{SOURCE}/charting", filename, refresh=False
                )
                (report.downloaded if downloaded else report.cached).append(filename)
            except HostBlocked as exc:
                report.blocked_hosts.add(exc.host)
                return report
            except DataUnavailable:
                report.missing.append(filename)
    return report


def fetch_all(
    tours: tuple[str, ...] = ("atp", "wta"),
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    charting: bool = True,
) -> list[FetchReport]:
    reports = [fetch_matches(tours, start_year, end_year), fetch_reference(tours)]
    if charting:
        reports.append(fetch_charting())
    return reports


def load_cached_matches(tour: str) -> pd.DataFrame:
    """Concatenate every cached season file for one tour."""
    directory = RAW_DIR / SOURCE / tour.lower()
    if not directory.exists():
        # Fall back to the flat layout used before sources were split up.
        directory = RAW_DIR / tour.lower()
    if not directory.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(directory.glob(f"{tour.lower()}_matches_*.csv")):
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except Exception as exc:
            log.warning("skipping unreadable %s: %s", path.name, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_charting_table(table: str, gender: str = "m") -> pd.DataFrame:
    path = cache_path(f"{SOURCE}/charting", f"charting-{gender}-{table}.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log.warning("unreadable charting table %s: %s", path.name, exc)
        return pd.DataFrame()


def charting_available() -> bool:
    return bool(list((RAW_DIR / SOURCE / "charting").glob("charting-*-stats-Overview.csv"))) if (
        RAW_DIR / SOURCE / "charting"
    ).exists() else False
