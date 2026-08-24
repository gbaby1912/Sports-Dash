"""Normalise raw archive CSVs into the canonical match table.

Output is one row per match with a winner/loser layout, cleaned types, parsed
scorelines, venue context and derived serve/return counting stats. Downstream
components never touch the raw files.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import LEVEL_WEIGHT, PROCESSED_DIR, SURFACES
from . import sources
from .score import parse_score
from .store import load_frame, save_frame
from .venues import is_indoor, venue_altitude

log = logging.getLogger(__name__)

_STAT_RENAME = {
    "ace": "ace",
    "df": "df",
    "svpt": "svpt",
    "1stIn": "first_in",
    "1stWon": "first_won",
    "2ndWon": "second_won",
    "SvGms": "sv_gms",
    "bpSaved": "bp_saved",
    "bpFaced": "bp_faced",
}

_PLAYER_RENAME = {
    "id": "id",
    "name": "name",
    "hand": "hand",
    "ht": "height_cm",
    "ioc": "ioc",
    "age": "age",
    "rank": "rank",
    "rank_points": "rank_points",
    "seed": "seed",
    "entry": "entry",
}


def _normalise_surface(value) -> str:
    if not isinstance(value, str):
        return "Hard"
    text = value.strip().title()
    if text in SURFACES:
        return text
    # A handful of rows carry blanks or oddities; Hard is the modal surface and
    # the safest default, but the row is flagged so it can be excluded.
    return "Unknown"


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan


def normalise_tour(raw: pd.DataFrame, tour: str) -> pd.DataFrame:
    """Convert one tour's raw archive frame into the canonical schema."""
    if raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    frame["tour"] = tour.lower()

    # --- dates -------------------------------------------------------------
    # tourney_date is an int like 20240513 and is the *start* of the event, not
    # the date of this particular match. We add the round offset later so that
    # chronological ordering within an event is right.
    frame["tourney_date"] = pd.to_datetime(
        frame["tourney_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    frame = frame[frame["tourney_date"].notna()].copy()

    # --- context -----------------------------------------------------------
    frame["surface"] = frame["surface"].map(_normalise_surface)
    frame["level"] = frame.get("tourney_level", "A").fillna("A").astype(str).str.strip().str[:1]
    frame["level_weight"] = frame["level"].map(LEVEL_WEIGHT).fillna(0.8)
    frame["round"] = frame.get("round", "R32").fillna("R32").astype(str)
    frame["best_of"] = pd.to_numeric(frame.get("best_of"), errors="coerce").fillna(3).astype(int)
    frame["draw_size"] = pd.to_numeric(frame.get("draw_size"), errors="coerce")
    frame["minutes"] = pd.to_numeric(frame.get("minutes"), errors="coerce")
    frame["match_num"] = pd.to_numeric(frame.get("match_num"), errors="coerce").fillna(0).astype(int)

    unique_events = frame[["tourney_name", "surface"]].drop_duplicates()
    indoor_map = {
        (name, surface): is_indoor(name, surface)
        for name, surface in unique_events.itertuples(index=False)
    }
    altitude_map = {name: venue_altitude(name) for name in frame["tourney_name"].dropna().unique()}
    frame["indoor"] = [
        indoor_map.get((n, s), False)
        for n, s in zip(frame["tourney_name"], frame["surface"])
    ]
    frame["altitude_m"] = frame["tourney_name"].map(altitude_map).fillna(0).astype(int)

    # --- player attributes -------------------------------------------------
    for side, prefix in (("winner", "winner"), ("loser", "loser")):
        for source_suffix, target in _PLAYER_RENAME.items():
            source = f"{side}_{source_suffix}"
            target_col = f"{prefix}_{target}"
            frame[target_col] = frame[source] if source in frame.columns else np.nan
        _to_numeric(frame, [f"{prefix}_{c}" for c in ("height_cm", "age", "rank", "rank_points")])
        frame[f"{prefix}_hand"] = (
            frame[f"{prefix}_hand"].fillna("U").astype(str).str.upper().str[:1].replace({"A": "L"})
        )

    # --- serve counting stats ---------------------------------------------
    for tag, prefix in (("w", "winner"), ("l", "loser")):
        for source_suffix, target in _STAT_RENAME.items():
            source = f"{tag}_{source_suffix}"
            frame[f"{prefix}_{target}"] = frame[source] if source in frame.columns else np.nan
        _to_numeric(frame, [f"{prefix}_{t}" for t in _STAT_RENAME.values()])

    # --- scoreline ---------------------------------------------------------
    parsed = [
        parse_score(score, best_of)
        for score, best_of in zip(frame["score"], frame["best_of"])
    ]
    frame["retirement"] = [p.retirement for p in parsed]
    frame["walkover"] = [p.walkover for p in parsed]
    frame["completed"] = [p.completed for p in parsed]
    frame["sets_played"] = [p.sets_played for p in parsed]
    frame["went_to_decider"] = [p.went_to_decider for p in parsed]
    frame["tiebreaks_played"] = [p.tiebreaks_played for p in parsed]
    frame["dominance"] = [p.dominance for p in parsed]
    frame["winner_games_won"] = [p.winner_games for p in parsed]
    frame["loser_games_won"] = [p.loser_games for p in parsed]
    frame["winner_sets_won"] = [p.winner_sets for p in parsed]
    frame["loser_sets_won"] = [p.loser_sets for p in parsed]
    frame["winner_tiebreaks_won"] = [p.winner_tiebreaks for p in parsed]
    frame["loser_tiebreaks_won"] = [
        p.tiebreaks_played - p.winner_tiebreaks for p in parsed
    ]

    # --- derived serve/return ---------------------------------------------
    # Serve points won is the pair (first_won + second_won). Return points won
    # is the complement of the opponent's serve points won - the archives do not
    # record it directly, but it is exactly recoverable.
    # Serve points won must exist for *both* players before the return columns,
    # which are defined as the complement of the opponent's serve.
    for prefix in ("winner", "loser"):
        frame[f"{prefix}_spw"] = frame[f"{prefix}_first_won"] + frame[f"{prefix}_second_won"]
    for prefix, other in (("winner", "loser"), ("loser", "winner")):
        frame[f"{prefix}_rtpt"] = frame[f"{other}_svpt"]
        frame[f"{prefix}_rpw"] = frame[f"{other}_svpt"] - frame[f"{other}_spw"]
        frame[f"{prefix}_bp_opps"] = frame[f"{other}_bp_faced"]
        frame[f"{prefix}_bp_conv"] = frame[f"{other}_bp_faced"] - frame[f"{other}_bp_saved"]

    for prefix in ("winner", "loser"):
        svpt = frame[f"{prefix}_svpt"]
        rtpt = frame[f"{prefix}_rtpt"]
        frame[f"{prefix}_spw_pct"] = np.where(svpt > 0, frame[f"{prefix}_spw"] / svpt, np.nan)
        frame[f"{prefix}_rpw_pct"] = np.where(rtpt > 0, frame[f"{prefix}_rpw"] / rtpt, np.nan)

    frame["has_serve_stats"] = (
        frame["winner_svpt"].gt(0).fillna(False)
        & frame["loser_svpt"].gt(0).fillna(False)
        & frame["winner_spw"].notna()
        & frame["loser_spw"].notna()
    )
    # Guard against impossible rows (more points won than played).
    impossible = (
        (frame["winner_spw"] > frame["winner_svpt"])
        | (frame["loser_spw"] > frame["loser_svpt"])
        | (frame["winner_spw"] < 0)
        | (frame["loser_spw"] < 0)
    )
    frame.loc[impossible.fillna(False), "has_serve_stats"] = False

    # --- ordering ----------------------------------------------------------
    frame["match_date"] = _match_date(frame)
    frame["match_id"] = (
        frame["tour"].astype(str)
        + "-"
        + frame["tourney_id"].astype(str)
        + "-"
        + frame["match_num"].astype(str)
    )
    frame = frame.drop_duplicates(subset=["match_id"], keep="first")
    frame = frame.sort_values(["match_date", "tourney_id", "match_num"], kind="mergesort")
    return frame.reset_index(drop=True)


def _match_date(frame: pd.DataFrame) -> pd.Series:
    """Approximate the true match date from the event start plus the round.

    The archives only give the tournament start date, so every match in an event
    shares a timestamp. That breaks chronological ordering *within* an event -
    a final would update ratings at the same instant as a first round. Adding a
    round-based day offset restores the correct causal order, which matters
    because ratings are updated match by match.
    """
    from .schema import ROUND_ORDER

    order = frame["round"].map(ROUND_ORDER).fillna(7)
    draw = frame["draw_size"].fillna(32)
    # Larger draws run longer, so each round step is worth more elapsed time.
    step = np.where(draw >= 96, 1.4, np.where(draw >= 48, 1.1, 0.9))
    offset = np.clip((order - 7) * step + 1, 0, 14)
    return frame["tourney_date"] + pd.to_timedelta(offset.round().astype(int), unit="D")


def build_match_table(
    tours: tuple[str, ...] = ("atp", "wta"),
    save: bool = True,
) -> pd.DataFrame:
    """Load every cached raw archive, normalise it, and persist the result."""
    frames = []
    for tour in tours:
        raw = sources.load_cached_matches(tour)
        if raw.empty:
            log.warning("no cached raw data for %s", tour)
            continue
        normalised = normalise_tour(raw, tour)
        log.info("%s: %d matches normalised", tour, len(normalised))
        frames.append(normalised)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["match_date", "tour", "tourney_id", "match_num"],
                                    kind="mergesort").reset_index(drop=True)
    if save:
        written = save_frame(combined, PROCESSED_DIR / "matches")
        log.info("wrote %s (%d rows)", written, len(combined))
    return combined


def load_match_table() -> pd.DataFrame:
    """Read the normalised match table from disk."""
    frame = load_frame(PROCESSED_DIR / "matches")
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    return frame
