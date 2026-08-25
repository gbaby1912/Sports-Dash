"""tennis-data.co.uk - the tour odds archive.

This is the source that makes the model checkable against reality. It publishes
one spreadsheet per season per tour, from 2000 to the present, and each row
carries what no results archive does:

* **Closing odds from several books**, including Pinnacle (the sharpest widely
  published line) and Bet365, plus the market maximum and average across books.
  A model's log loss is an internal number; whether it beats the closing line is
  the only external verdict that cannot be gamed by picking a friendly metric.
* **A native Indoor/Outdoor flag.** The Sackmann archives do not carry this and
  it has to be curated by tournament name; here it is a column.
* **Series/Tier, set-by-set scores, both players' rank and ranking points, and a
  completion status** that distinguishes Completed from Retired and Walkover.

What it does *not* carry is serve statistics - no aces, no service points, no
break points. That is why this source is joined to the Sackmann archives rather
than replacing them: odds and context from here, point-level detail from there.

File layout (stable since 2001):
    ATP   http://www.tennis-data.co.uk/{year}/{year}.xlsx
    WTA   http://www.tennis-data.co.uk/{year}w/{year}.xlsx
Early seasons are published as .zip containing .xls, so both are handled.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import SURFACES
from .http import DataUnavailable, FetchReport, HostBlocked, cache_path, download_cached

log = logging.getLogger(__name__)

BASE_URL = "http://www.tennis-data.co.uk"
HOST = "www.tennis-data.co.uk"
SOURCE = "tennis_data"
FIRST_SEASON = 2000

# Bookmaker column prefixes, in descending order of how much we trust the line.
# Pinnacle first: it is the low-margin, high-limit book whose closing price is
# the standard benchmark. Max/Avg are derived across all books in the file.
BOOKMAKERS = [
    ("ps", "PS", "Pinnacle"),
    ("b365", "B365", "Bet365"),
    ("ex", "EX", "Expekt"),
    ("lb", "LB", "Ladbrokes"),
    ("sj", "SJ", "Stan James"),
    ("ub", "UB", "Unibet"),
    ("cb", "CB", "Centrebet"),
    ("gb", "GB", "Gamebookers"),
    ("iw", "IW", "Interwetten"),
    ("sb", "SB", "Sportingbet"),
    ("bw", "B&W", "Bet&Win"),
]

DERIVED_PRICES = [("max", "Max"), ("avg", "Avg")]

# tennis-data's Series/Tier vocabulary mapped onto the archive's level codes,
# so both sources describe an event the same way.
SERIES_TO_LEVEL = {
    "grand slam": "G",
    "masters 1000": "M",
    "masters": "M",
    "masters cup": "F",
    "atp500": "A",
    "atp250": "A",
    "international gold": "A",
    "international": "A",
    "premier mandatory": "M",
    "premier m": "M",
    "premier": "M",
    "premier 5": "M",
    "wta1000": "M",
    "wta500": "A",
    "wta250": "A",
    "tier i": "M",
    "tier ii": "A",
    "tier iii": "A",
    "tier iv": "A",
    "tier v": "A",
    "grand prix": "A",
}


def season_url(tour: str, year: int) -> str:
    suffix = "" if tour.lower() == "atp" else "w"
    return f"{BASE_URL}/{year}{suffix}/{year}.xlsx"


def season_zip_url(tour: str, year: int) -> str:
    suffix = "" if tour.lower() == "atp" else "w"
    return f"{BASE_URL}/{year}{suffix}/{year}.zip"


def fetch(
    tours: tuple[str, ...] = ("atp", "wta"),
    start_year: int = FIRST_SEASON,
    end_year: int | None = None,
    refresh_current: bool = True,
) -> FetchReport:
    """Download every season spreadsheet into the raw cache."""
    end_year = end_year or pd.Timestamp.today().year
    report = FetchReport(source=SOURCE)

    for tour in tours:
        for year in range(start_year, end_year + 1):
            filename = f"{tour}_{year}.xlsx"
            refresh = refresh_current and year >= end_year
            try:
                path, downloaded = download_cached(
                    season_url(tour, year), SOURCE, filename, refresh=refresh
                )
                (report.downloaded if downloaded else report.cached).append(filename)
                continue
            except HostBlocked as exc:
                report.blocked_hosts.add(exc.host)
                return report  # no point continuing against a blocked host
            except DataUnavailable:
                pass

            # Older seasons ship as a zip of .xls.
            try:
                path, downloaded = download_cached(
                    season_zip_url(tour, year), SOURCE, f"{tour}_{year}.zip", refresh=refresh
                )
                (report.downloaded if downloaded else report.cached).append(f"{tour}_{year}.zip")
            except HostBlocked as exc:
                report.blocked_hosts.add(exc.host)
                return report
            except DataUnavailable as exc:
                report.missing.append(f"{tour} {year}")
                log.debug("tennis-data %s %s: %s", tour, year, exc)

    return report


def _read_spreadsheet(path: Path) -> pd.DataFrame:
    """Read an .xlsx, or the first sheet of an .xls inside a .zip."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith((".xls", ".xlsx"))]
            if not names:
                raise DataUnavailable(f"no spreadsheet inside {path.name}")
            payload = archive.read(names[0])
            engine = "openpyxl" if names[0].lower().endswith(".xlsx") else "xlrd"
            return pd.read_excel(io.BytesIO(payload), engine=engine)
    return pd.read_excel(path, engine="openpyxl")


def _normalise_court(value) -> bool | None:
    if not isinstance(value, str):
        return None
    return value.strip().lower().startswith("indoor")


def _normalise_surface(value) -> str:
    if not isinstance(value, str):
        return "Unknown"
    text = value.strip().title()
    return text if text in SURFACES else "Unknown"


def _completion(comment) -> tuple[bool, bool]:
    """(retirement, walkover) from the Comment column."""
    if not isinstance(comment, str):
        return False, False
    text = comment.strip().lower()
    return text.startswith("retired"), text.startswith(("walkover", "w/o", "disq"))


def load(tours: tuple[str, ...] = ("atp", "wta")) -> pd.DataFrame:
    """Read every cached season file into one canonical odds frame."""
    frames = []
    for tour in tours:
        directory = cache_path(SOURCE, ".").parent
        for path in sorted(directory.glob(f"{tour}_*.xls*")) + sorted(
            directory.glob(f"{tour}_*.zip")
        ):
            try:
                raw = _read_spreadsheet(path)
            except Exception as exc:
                log.warning("skipping unreadable %s: %s", path.name, exc)
                continue
            frames.append(normalise(raw, tour))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["match_date"].notna()]
    return combined.sort_values("match_date", kind="mergesort").reset_index(drop=True)


def normalise(raw: pd.DataFrame, tour: str) -> pd.DataFrame:
    """Convert one raw season spreadsheet into the canonical odds schema."""
    if raw.empty:
        return pd.DataFrame()

    frame = pd.DataFrame()
    columns = {c.strip().lower(): c for c in raw.columns.astype(str)}

    def column(*names, default=None):
        for name in names:
            key = name.strip().lower()
            if key in columns:
                return raw[columns[key]]
        return pd.Series([default] * len(raw), index=raw.index)

    frame["tour"] = tour.lower()
    frame["match_date"] = pd.to_datetime(column("date"), errors="coerce")
    frame["tournament"] = column("tournament").astype(str).str.strip()
    frame["location"] = column("location").astype(str).str.strip()
    series = column("series", "tier").astype(str).str.strip()
    frame["series"] = series
    frame["level"] = series.str.lower().map(SERIES_TO_LEVEL).fillna("A")
    frame["indoor"] = column("court").map(_normalise_court)
    frame["surface"] = column("surface").map(_normalise_surface)
    frame["round"] = column("round").astype(str).str.strip()
    frame["best_of"] = pd.to_numeric(column("best of"), errors="coerce").fillna(3).astype(int)

    frame["winner_name"] = column("winner").astype(str).str.strip()
    frame["loser_name"] = column("loser").astype(str).str.strip()
    frame["winner_rank"] = pd.to_numeric(column("wrank"), errors="coerce")
    frame["loser_rank"] = pd.to_numeric(column("lrank"), errors="coerce")
    frame["winner_rank_points"] = pd.to_numeric(column("wpts"), errors="coerce")
    frame["loser_rank_points"] = pd.to_numeric(column("lpts"), errors="coerce")

    # Set scores, so the join can be verified against the results archive.
    for set_number in range(1, 6):
        frame[f"w{set_number}"] = pd.to_numeric(column(f"w{set_number}"), errors="coerce")
        frame[f"l{set_number}"] = pd.to_numeric(column(f"l{set_number}"), errors="coerce")
    frame["winner_sets"] = pd.to_numeric(column("wsets"), errors="coerce")
    frame["loser_sets"] = pd.to_numeric(column("lsets"), errors="coerce")

    comment = column("comment")
    completion = [_completion(c) for c in comment]
    frame["retirement"] = [c[0] for c in completion]
    frame["walkover"] = [c[1] for c in completion]
    frame["comment"] = comment.astype(str)

    # --- odds -------------------------------------------------------------
    for key, prefix, _label in BOOKMAKERS:
        winner = pd.to_numeric(column(f"{prefix}W"), errors="coerce")
        loser = pd.to_numeric(column(f"{prefix}L"), errors="coerce")
        if winner.notna().any() or loser.notna().any():
            frame[f"odds_w_{key}"] = _sanitise_odds(winner)
            frame[f"odds_l_{key}"] = _sanitise_odds(loser)
    for key, prefix in DERIVED_PRICES:
        frame[f"odds_w_{key}"] = _sanitise_odds(
            pd.to_numeric(column(f"{prefix}W"), errors="coerce")
        )
        frame[f"odds_l_{key}"] = _sanitise_odds(
            pd.to_numeric(column(f"{prefix}L"), errors="coerce")
        )

    frame["source"] = SOURCE
    return frame


def _sanitise_odds(values: pd.Series) -> pd.Series:
    """Decimal odds must exceed 1.0; anything else is a data error, not a price."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where((numeric > 1.0) & (numeric < 1000.0))


def best_available_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-book columns into one preferred price pair.

    Preference order is Pinnacle, then the cross-book average, then Bet365, then
    whatever else is present. Pinnacle first because its low margin makes it the
    closest thing to a consensus true price; the average is the next best proxy
    when Pinnacle is missing, and a single retail book is the last resort.
    """
    preference = ["ps", "avg", "b365", "max"] + [k for k, _, _ in BOOKMAKERS]
    winner = pd.Series(np.nan, index=frame.index, dtype=float)
    loser = pd.Series(np.nan, index=frame.index, dtype=float)
    book = pd.Series(pd.NA, index=frame.index, dtype="object")

    for key in preference:
        w_col, l_col = f"odds_w_{key}", f"odds_l_{key}"
        if w_col not in frame.columns or l_col not in frame.columns:
            continue
        usable = winner.isna() & frame[w_col].notna() & frame[l_col].notna()
        winner = winner.mask(usable, frame[w_col])
        loser = loser.mask(usable, frame[l_col])
        book = book.mask(usable, key)

    out = frame.copy()
    out["odds_winner"] = winner
    out["odds_loser"] = loser
    out["odds_book"] = book
    return out
