"""Elo rating family.

Four ratings are maintained per player, each answering a different question:

``overall``
    Classic match-outcome Elo. The strongest single baseline in tennis.
``surface``
    Outcome Elo restricted to one surface. Clay and grass specialists are real,
    and a player's clay rating can sit 200+ points from their hard rating.
``points``
    Elo fitted to the *share of total points won* rather than the binary result.
    A 7-6 7-6 win and a 6-0 6-0 win are the same event to outcome Elo; they are
    very different evidence. Because a match is ~150 points rather than one
    binary trial, this rating has far lower variance and reacts faster to a
    genuine change in level.
``games``
    Same idea one level up, on share of games won. Slightly noisier than points
    but available for the historical matches that lack point-level stats.

All four are updated strictly in chronological order and every feature is read
*before* the match is applied, so there is no look-ahead leakage by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CONFIG, EloConfig

RATING_KINDS = ("overall", "surface", "points", "games")


@dataclass
class PlayerState:
    rating: float
    matches: int = 0
    last_date: pd.Timestamp | None = None


@dataclass
class EloEngine:
    """Chronological multi-rating Elo engine."""

    config: EloConfig = field(default_factory=lambda: CONFIG.elo)

    def __post_init__(self) -> None:
        self._overall: dict[tuple[str, int], PlayerState] = {}
        self._surface: dict[tuple[str, int, str], PlayerState] = {}
        self._points: dict[tuple[str, int], PlayerState] = {}
        self._games: dict[tuple[str, int], PlayerState] = {}
        # A plain dict, not a defaultdict with a closure: the engine has to
        # survive joblib serialisation, and a lambda default_factory is not
        # picklable.
        self._peak: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------- internals
    def _state(self, store: dict, key) -> PlayerState:
        if key not in store:
            store[key] = PlayerState(rating=self.config.initial)
        return store[key]

    def _k_factor(self, matches: int, surface: bool = False) -> float:
        cfg = self.config
        k = cfg.k_scale / (matches + cfg.k_shift) ** cfg.k_decay
        return k * (cfg.surface_k_multiplier if surface else 1.0)

    def _apply_layoff(self, state: PlayerState, date: pd.Timestamp) -> None:
        """Regress a stale rating toward the mean.

        A rating is a claim about current level. After a long absence - injury,
        maternity, suspension - that claim is much weaker, and players
        overwhelmingly return below their old level. Regressing toward the mean
        encodes that uncertainty instead of pretending nothing happened.
        """
        cfg = self.config
        if state.last_date is None:
            return
        idle_days = (date - state.last_date).days
        if idle_days <= cfg.layoff_grace_days:
            return
        years_out = (idle_days - cfg.layoff_grace_days) / 365.25
        shrink = min(cfg.layoff_regression_cap, cfg.layoff_regression_per_year * years_out)
        state.rating += (cfg.initial - state.rating) * shrink

    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _margin_multiplier(self, dominance: float) -> float:
        cfg = self.config
        if not cfg.use_margin_of_victory or not np.isfinite(dominance):
            return 1.0
        spread = abs(dominance - 0.5) * 2.0
        return cfg.mov_min + (cfg.mov_max - cfg.mov_min) * float(np.clip(spread, 0.0, 1.0))

    # ---------------------------------------------------------------- public
    def rating_of(self, tour: str, player_id: int, kind: str = "overall",
                  surface: str | None = None) -> float:
        if kind == "surface":
            key = (tour, player_id, surface or "Hard")
            return self._surface[key].rating if key in self._surface else self.config.initial
        store = {"overall": self._overall, "points": self._points, "games": self._games}[kind]
        key = (tour, player_id)
        return store[key].rating if key in store else self.config.initial

    def matches_played(self, tour: str, player_id: int) -> int:
        key = (tour, player_id)
        return self._overall[key].matches if key in self._overall else 0

    def blended(self, tour: str, player_id: int, surface: str) -> float:
        """Surface rating shrunk toward the overall rating.

        Pure surface Elo is badly under-powered: a player may have only 15 grass
        matches in their career. Blending with overall Elo keeps the surface
        signal while borrowing strength from the full record. The blend weight
        is higher on clay and grass, which are the most idiosyncratic surfaces.
        """
        weight = self.config.surface_blend.get(surface, 0.6)
        surface_state = self._surface.get((tour, player_id, surface))
        overall = self.rating_of(tour, player_id, "overall")
        if surface_state is None:
            return overall
        # Additionally shrink toward overall when the surface sample is thin.
        confidence = surface_state.matches / (surface_state.matches + 12.0)
        effective = weight * confidence
        return effective * surface_state.rating + (1.0 - effective) * overall

    def peak(self, tour: str, player_id: int) -> float:
        return self._peak.get((tour, player_id), self.config.initial)

    def pre_match_features(self, row) -> dict[str, float]:
        """Ratings for both players as they stand *before* this match."""
        tour, surface = row["tour"], row["surface"]
        winner, loser = int(row["winner_id"]), int(row["loser_id"])
        out: dict[str, float] = {}
        for tag, player in (("w", winner), ("l", loser)):
            out[f"{tag}_elo"] = self.rating_of(tour, player, "overall")
            out[f"{tag}_elo_surface"] = self.rating_of(tour, player, "surface", surface)
            out[f"{tag}_elo_blend"] = self.blended(tour, player, surface)
            out[f"{tag}_elo_points"] = self.rating_of(tour, player, "points")
            out[f"{tag}_elo_games"] = self.rating_of(tour, player, "games")
            out[f"{tag}_elo_peak"] = self.peak(tour, player)
            out[f"{tag}_elo_matches"] = float(self.matches_played(tour, player))
            surface_state = self._surface.get((tour, player, surface))
            out[f"{tag}_elo_surface_matches"] = float(surface_state.matches) if surface_state else 0.0
        return out

    def update(self, row) -> None:
        """Apply one match result to every rating."""
        cfg = self.config
        tour, surface = row["tour"], row["surface"]
        date = row["match_date"]
        winner, loser = int(row["winner_id"]), int(row["loser_id"])

        weight = 1.0
        if bool(row.get("walkover", False)):
            weight = cfg.walkover_weight
        elif bool(row.get("retirement", False)):
            weight = cfg.retirement_weight
        if cfg.use_level_weight:
            weight *= float(row.get("level_weight", 1.0) or 1.0)
        if weight <= 0:
            # Still record that the players were active, so layoff logic is right.
            for store, key in (
                (self._overall, (tour, winner)), (self._overall, (tour, loser)),
            ):
                self._state(store, key).last_date = date
            return

        dominance = float(row.get("dominance", 0.5) or 0.5)
        mov = self._margin_multiplier(dominance)

        # --- outcome ratings (overall + surface) ---------------------------
        self._update_pair(
            self._overall, (tour, winner), (tour, loser),
            actual=1.0, date=date, weight=weight * mov, surface=False,
        )
        self._update_pair(
            self._surface, (tour, winner, surface), (tour, loser, surface),
            actual=1.0, date=date, weight=weight * mov, surface=True,
        )

        # --- margin ratings ------------------------------------------------
        # Point share is only available when the archive carries serve stats.
        if bool(row.get("has_serve_stats", False)):
            w_points = float(row["winner_spw"]) + float(row["winner_rpw"])
            total_points = w_points + float(row["loser_spw"]) + float(row["loser_rpw"])
            if total_points > 0:
                share = w_points / total_points
                self._update_pair(
                    self._points, (tour, winner), (tour, loser),
                    actual=_stretch(share), date=date, weight=weight, surface=False,
                )

        total_games = float(row.get("winner_games_won", 0)) + float(row.get("loser_games_won", 0))
        if total_games > 0:
            game_share = float(row["winner_games_won"]) / total_games
            self._update_pair(
                self._games, (tour, winner), (tour, loser),
                actual=_stretch(game_share), date=date, weight=weight, surface=False,
            )

        for player in (winner, loser):
            key = (tour, player)
            self._peak[key] = max(
                self._peak.get(key, self.config.initial), self._overall[key].rating
            )

    def _update_pair(
        self,
        store: dict,
        key_a,
        key_b,
        actual: float,
        date: pd.Timestamp,
        weight: float,
        surface: bool,
    ) -> None:
        state_a, state_b = self._state(store, key_a), self._state(store, key_b)
        self._apply_layoff(state_a, date)
        self._apply_layoff(state_b, date)

        expected = self.expected(state_a.rating, state_b.rating)
        k_a = self._k_factor(state_a.matches, surface) * weight
        k_b = self._k_factor(state_b.matches, surface) * weight
        delta = actual - expected

        state_a.rating += k_a * delta
        state_b.rating -= k_b * delta
        state_a.matches += 1
        state_b.matches += 1
        state_a.last_date = date
        state_b.last_date = date

    def snapshot(
        self,
        tour: str,
        surface: str | None = None,
        active_since: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Current ratings for every player on a tour, for leaderboards.

        ``active_since`` drops players who have not competed since that date. A
        leaderboard without it is misleading rather than merely stale: Elo only
        decays on a *played* match, so a player who retired years ago keeps the
        rating they walked away with and can sit above everyone currently on tour.
        """
        rows = []
        for (t, player_id), state in self._overall.items():
            if t != tour:
                continue
            if active_since is not None and (
                state.last_date is None or state.last_date < active_since
            ):
                continue
            row = {
                "player_id": player_id,
                "elo": state.rating,
                "elo_points": self.rating_of(tour, player_id, "points"),
                "elo_games": self.rating_of(tour, player_id, "games"),
                "matches": state.matches,
                "peak_elo": self.peak(tour, player_id),
                "last_date": state.last_date,
            }
            for surf in CONFIG.elo.surface_blend:
                row[f"elo_{surf.lower()}"] = self.blended(tour, player_id, surf)
            rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        sort_key = f"elo_{surface.lower()}" if surface else "elo"
        return frame.sort_values(sort_key, ascending=False).reset_index(drop=True)


def _stretch(share: float, strength: float = 2.6) -> float:
    """Map a points/games share onto an Elo-style [0, 1] performance score.

    Point shares live in a narrow band - winning 55% of total points is a rout -
    so feeding the raw share to an Elo update would barely move ratings. This
    stretches the band around 0.5 while keeping the result inside (0, 1) and
    exactly symmetric, so ``_stretch(x) + _stretch(1-x) == 1``.
    """
    centered = (float(share) - 0.5) * strength
    return float(np.clip(0.5 + centered, 0.02, 0.98))


def run_elo(matches: pd.DataFrame, config: EloConfig | None = None) -> tuple[pd.DataFrame, EloEngine]:
    """Stream every match through the engine, emitting pre-match ratings.

    Returns the per-match rating features and the fitted engine (whose final
    state is the current rating table used by the dashboard).
    """
    engine = EloEngine(config or CONFIG.elo)
    records: list[dict[str, float]] = []
    ordered = matches.sort_values(["match_date", "tour", "tourney_id", "match_num"], kind="mergesort")
    for row in ordered.to_dict("records"):
        records.append(engine.pre_match_features(row))
        engine.update(row)
    frame = pd.DataFrame(records, index=ordered.index)
    return frame.reindex(matches.index), engine
