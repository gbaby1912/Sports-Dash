"""Player directory: the identity and attribute lookup the dashboard needs.

Built from the match archive rather than the separate player table, so it works
for any data source that carries the standard match columns and always reflects
the players who actually appear in the modelled data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_directory(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (tour, player) with their most recent known attributes."""
    frames = []
    for prefix in ("winner", "loser"):
        block = matches[[
            "tour", "match_date", f"{prefix}_id", f"{prefix}_name", f"{prefix}_hand",
            f"{prefix}_height_cm", f"{prefix}_ioc", f"{prefix}_age",
            f"{prefix}_rank", f"{prefix}_rank_points",
        ]].copy()
        block.columns = [
            "tour", "match_date", "player_id", "name", "hand",
            "height_cm", "ioc", "age", "rank", "rank_points",
        ]
        block["won"] = 1 if prefix == "winner" else 0
        frames.append(block)

    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked[stacked["player_id"].notna()]
    stacked["player_id"] = stacked["player_id"].astype(np.int64)
    stacked = stacked.sort_values("match_date", kind="mergesort")

    aggregated = stacked.groupby(["tour", "player_id"]).agg(
        name=("name", "last"),
        hand=("hand", "last"),
        height_cm=("height_cm", "last"),
        ioc=("ioc", "last"),
        last_age=("age", "last"),
        last_rank=("rank", "last"),
        last_rank_points=("rank_points", "last"),
        last_played=("match_date", "max"),
        first_played=("match_date", "min"),
        matches=("won", "size"),
        wins=("won", "sum"),
    ).reset_index()
    aggregated["win_pct"] = aggregated["wins"] / aggregated["matches"].clip(lower=1)
    return aggregated


def age_on(directory_row: pd.Series, date: pd.Timestamp) -> float:
    """Extrapolate a player's age forward from their last recorded match."""
    last_age = directory_row.get("last_age")
    last_played = directory_row.get("last_played")
    if pd.isna(last_age):
        return float("nan")
    if pd.isna(last_played):
        return float(last_age)
    years = (pd.Timestamp(date) - pd.Timestamp(last_played)).days / 365.25
    return float(last_age) + float(max(years, 0.0))
