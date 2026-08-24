"""Canonical match schema shared by every downstream component.

The public archives (Jeff Sackmann's ``tennis_atp`` / ``tennis_wta``) use a
winner/loser layout. We keep that layout in the *normalised* table because it is
lossless, and only re-orient to a symmetric ``p1``/``p2`` layout when the
feature matrix is built. That keeps the ingestion stage dumb and testable.
"""
from __future__ import annotations

# Columns copied verbatim (after renaming) from the source archives.
IDENTITY_COLUMNS = [
    "match_id",
    "tour",
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "match_num",
    "surface",
    "indoor",
    "altitude_m",
    "draw_size",
    "level",
    "round",
    "best_of",
    "minutes",
    "score",
]

PLAYER_COLUMNS = [
    "id",
    "name",
    "hand",
    "height_cm",
    "ioc",
    "age",
    "rank",
    "rank_points",
    "seed",
    "entry",
]

# Raw per-match serve counting stats, as recorded in the archives.
SERVE_STAT_COLUMNS = [
    "ace",
    "df",
    "svpt",       # service points played
    "first_in",   # first serves in
    "first_won",  # points won behind first serve
    "second_won", # points won behind second serve
    "sv_gms",     # service games played
    "bp_saved",
    "bp_faced",
]

# Columns derived during normalisation.
DERIVED_MATCH_COLUMNS = [
    "retirement",
    "walkover",
    "completed",
    "sets_played",
    "went_to_decider",
    "tiebreaks_played",
    "level_weight",
]

DERIVED_PLAYER_COLUMNS = [
    "spw",        # serve points won
    "rpw",        # return points won
    "rtpt",       # return points played
    "spw_pct",
    "rpw_pct",
    "games_won",
    "sets_won",
    "tiebreaks_won",
    "bp_conv",    # break points converted (== opponent bp_faced - bp_saved)
    "bp_opps",    # break point opportunities (== opponent bp_faced)
]

ROUND_ORDER = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3,
    "ER": 4, "BR": 5, "RR": 6,
    "R128": 7, "R64": 8, "R32": 9, "R16": 10,
    "QF": 11, "SF": 12, "F": 13,
}


def player_columns(prefix: str) -> list[str]:
    """Fully-qualified column names for one side of a match."""
    cols = [f"{prefix}_{c}" for c in PLAYER_COLUMNS]
    cols += [f"{prefix}_{c}" for c in SERVE_STAT_COLUMNS]
    cols += [f"{prefix}_{c}" for c in DERIVED_PLAYER_COLUMNS]
    return cols


NORMALISED_COLUMNS = (
    IDENTITY_COLUMNS
    + DERIVED_MATCH_COLUMNS
    + player_columns("winner")
    + player_columns("loser")
)
