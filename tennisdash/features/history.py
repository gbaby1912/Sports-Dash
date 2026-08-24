"""Rolling player-history features: form, fatigue, clutch, streaks, head-to-head.

Everything here is computed in a single chronological pass. Each match reads the
state accumulated from strictly earlier matches, then contributes to that state.
That ordering is the whole ballgame: it is very easy to build a tennis model
that scores 80% accuracy by accidentally letting a season-long average peek at
the match it is predicting.

The features fall into five groups.

**Form** - exponentially-decayed recent results. Weighting by opponent strength
matters: beating the world #3 and beating a qualifier are not the same evidence,
so wins are credited by the opponent's rating rather than counted.

**Fatigue and rest** - court time in the last 7/14/28 days, matches played,
five-setters survived, and days since the last match. Both ends hurt: a player
three days off a four-hour semi-final is depleted, and a player six weeks idle
is rusty. The relationship is U-shaped, so raw "days since last match" is given
to the model alongside an explicit distance-from-optimal term.

**Clutch** - break points saved and converted, tiebreak record, deciding-set
record. These are the most over-interpreted numbers in tennis: single-season
clutch splits are dominated by noise. They are included, but shrunk hard toward
tour average with explicit pseudo-counts so the model sees a stabilised estimate
rather than small-sample noise.

**Surface transition** - how much of the recent schedule was on this surface.
The week after a clay swing ends, results on grass are systematically noisier.

**Head-to-head** - shrunk toward the neutral prior. H2H is the single most
over-weighted factor in tennis punditry: most pairs have played fewer than four
times, and once you control for the rating gap at the time, historical H2H adds
very little. It is included with a heavy prior so the model can use it where a
genuinely long rivalry exists.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CONFIG, FeatureConfig

# Tour-average anchors used as shrinkage targets for the clutch metrics.
def _results_deque() -> deque:
    return deque(maxlen=120)


def _workload_deque() -> deque:
    return deque(maxlen=60)


def _surface_deque() -> deque:
    return deque(maxlen=40)


def _retirement_deque() -> deque:
    return deque(maxlen=10)


def _h2h_pair() -> list:
    return [0.0, 0.0]


TOUR_PRIORS = {
    "atp": {"bp_saved": 0.615, "bp_conv": 0.405, "tiebreak": 0.50, "decider": 0.50},
    "wta": {"bp_saved": 0.565, "bp_conv": 0.455, "tiebreak": 0.50, "decider": 0.50},
}


@dataclass
class PlayerHistory:
    """Mutable rolling state for one player on one tour."""

    # (date, won, opponent_elo, weight) for decayed form.
    results: deque = field(default_factory=_results_deque)
    # (date, minutes, sets) for fatigue.
    workload: deque = field(default_factory=_workload_deque)
    # (date, surface) for surface-transition features.
    surfaces: deque = field(default_factory=_surface_deque)

    last_date: pd.Timestamp | None = None
    matches: int = 0
    wins: int = 0
    win_streak: int = 0
    loss_streak: int = 0

    bp_saved: float = 0.0
    bp_faced: float = 0.0
    bp_conv: float = 0.0
    bp_opps: float = 0.0
    tiebreaks_won: float = 0.0
    tiebreaks_played: float = 0.0
    deciders_won: float = 0.0
    deciders_played: float = 0.0
    retirements: int = 0
    recent_retirements: deque = field(default_factory=_retirement_deque)

    # Per-surface record.
    surface_wins: dict = field(default_factory=lambda: defaultdict(float))
    surface_matches: dict = field(default_factory=lambda: defaultdict(float))


class HistoryEngine:
    """Single-pass rolling feature builder."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or CONFIG.features
        self.players: dict[tuple[str, int], PlayerHistory] = {}
        self.h2h: dict[tuple[str, int, int], list[float]] = defaultdict(_h2h_pair)
        self.h2h_surface: dict[tuple[str, int, int, str], list[float]] = defaultdict(_h2h_pair)

    def _get(self, tour: str, player: int) -> PlayerHistory:
        key = (tour, player)
        if key not in self.players:
            self.players[key] = PlayerHistory()
        return self.players[key]

    # ------------------------------------------------------------- features
    def _form(self, state: PlayerHistory, date: pd.Timestamp) -> dict[str, float]:
        cfg = self.config
        out: dict[str, float] = {}
        if not state.results:
            for window in cfg.form_windows:
                out[f"form_win_pct_{window}"] = np.nan
            out["form_decayed"] = np.nan
            out["form_quality"] = np.nan
            return out

        results = list(state.results)
        for window in cfg.form_windows:
            recent = results[-window:]
            out[f"form_win_pct_{window}"] = float(np.mean([r[1] for r in recent]))

        # Exponentially decayed win rate: last month counts far more than last year.
        ages = np.array([(date - r[0]).days for r in results], dtype=float)
        decay = 0.5 ** (ages / cfg.form_half_life_days)
        won = np.array([r[1] for r in results], dtype=float)
        total = decay.sum()
        out["form_decayed"] = float((decay * won).sum() / total) if total > 0 else np.nan

        # Quality-weighted form: credit for *who* was beaten, not just how many.
        opponent = np.array([r[2] for r in results], dtype=float)
        # Centre opponent strength so the number is "rating beaten, net of 1500".
        signed = np.where(won > 0.5, opponent - 1500.0, -(1500.0 - opponent + 200.0))
        out["form_quality"] = float((decay * signed).sum() / total) if total > 0 else np.nan
        return out

    def _fatigue(self, state: PlayerHistory, date: pd.Timestamp) -> dict[str, float]:
        cfg = self.config
        out: dict[str, float] = {}
        workload = [(d, m, s) for d, m, s in state.workload]
        for days in cfg.fatigue_windows_days:
            cutoff = date - pd.Timedelta(days=days)
            window = [(d, m, s) for d, m, s in workload if d >= cutoff]
            out[f"minutes_{days}d"] = float(sum(m for _, m, _ in window if np.isfinite(m)))
            out[f"matches_{days}d"] = float(len(window))
            out[f"sets_{days}d"] = float(sum(s for _, _, s in window if np.isfinite(s)))

        if state.last_date is None:
            out["days_since_last"] = np.nan
            out["rest_deviation"] = np.nan
            out["is_returning"] = 1.0
        else:
            idle = float((date - state.last_date).days)
            out["days_since_last"] = idle
            # Distance from the ideal amount of rest, in log space so that
            # 2 days and 40 days both read as "far from optimal" without one
            # week of layoff dominating the scale.
            out["rest_deviation"] = float(
                abs(np.log1p(idle) - np.log1p(cfg.rest_optimal_days))
            )
            out["is_returning"] = 1.0 if idle > 120 else 0.0

        # A recent retirement is the best public proxy for carrying an injury.
        cutoff = date - pd.Timedelta(days=90)
        out["recent_retirements"] = float(sum(1 for d in state.recent_retirements if d >= cutoff))
        return out

    def _clutch(self, state: PlayerHistory, tour: str) -> dict[str, float]:
        cfg = self.config
        priors = TOUR_PRIORS.get(tour, TOUR_PRIORS["atp"])

        def shrink(made: float, total: float, prior_rate: float, prior_n: float) -> float:
            return (made + prior_rate * prior_n) / (total + prior_n)

        return {
            "bp_saved_pct": shrink(state.bp_saved, state.bp_faced, priors["bp_saved"], cfg.bp_prior),
            "bp_conv_pct": shrink(state.bp_conv, state.bp_opps, priors["bp_conv"], cfg.bp_prior),
            "tiebreak_pct": shrink(
                state.tiebreaks_won, state.tiebreaks_played, priors["tiebreak"], cfg.tiebreak_prior
            ),
            "decider_pct": shrink(
                state.deciders_won, state.deciders_played, priors["decider"], cfg.decider_prior
            ),
            "bp_faced_sample": float(state.bp_faced),
        }

    def _surface_context(
        self, state: PlayerHistory, surface: str, date: pd.Timestamp
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        played = state.surface_matches.get(surface, 0.0)
        wins = state.surface_wins.get(surface, 0.0)
        # Shrink the surface win rate toward the player's overall win rate.
        overall = state.wins / state.matches if state.matches else 0.5
        out["surface_win_pct"] = (wins + 8.0 * overall) / (played + 8.0)
        out["surface_matches"] = float(played)

        cutoff = date - pd.Timedelta(days=60)
        recent = [s for d, s in state.surfaces if d >= cutoff]
        if recent:
            out["surface_recency"] = float(np.mean([s == surface for s in recent]))
        else:
            out["surface_recency"] = np.nan
        return out

    def _h2h(self, tour: str, a: int, b: int, surface: str) -> dict[str, float]:
        cfg = self.config
        prior = cfg.h2h_prior_matches
        # `.get` rather than `[...]`: these are defaultdicts, so indexing them
        # *inserts* an empty record. The engine is long-lived inside the API, and
        # a read that writes would grow it without bound as users explore
        # matchups that have never been played.
        wins, losses = self.h2h.get((tour, a, b), (0.0, 0.0))
        total = wins + losses
        surface_wins, surface_losses = self.h2h_surface.get((tour, a, b, surface), (0.0, 0.0))
        surface_total = surface_wins + surface_losses
        return {
            "h2h_matches": float(total),
            # Shrunk toward 0.5: with 2 prior meetings this barely moves.
            "h2h_win_pct": float((wins + 0.5 * prior) / (total + prior)),
            "h2h_surface_matches": float(surface_total),
            "h2h_surface_win_pct": float(
                (surface_wins + 0.5 * prior) / (surface_total + prior)
            ),
        }

    def pre_match_features(self, row: dict) -> dict[str, float]:
        """All rolling features for both players, as of just before this match."""
        tour, surface, date = row["tour"], row["surface"], row["match_date"]
        winner, loser = int(row["winner_id"]), int(row["loser_id"])
        out: dict[str, float] = {}
        for tag, player in (("w", winner), ("l", loser)):
            state = self._get(tour, player)
            for name, value in self._form(state, date).items():
                out[f"{tag}_{name}"] = value
            for name, value in self._fatigue(state, date).items():
                out[f"{tag}_{name}"] = value
            for name, value in self._clutch(state, tour).items():
                out[f"{tag}_{name}"] = value
            for name, value in self._surface_context(state, surface, date).items():
                out[f"{tag}_{name}"] = value
            out[f"{tag}_career_matches"] = float(state.matches)
            out[f"{tag}_career_win_pct"] = (
                state.wins / state.matches if state.matches else np.nan
            )
            out[f"{tag}_win_streak"] = float(state.win_streak)
            out[f"{tag}_loss_streak"] = float(state.loss_streak)

        for tag, a, b in (("w", winner, loser), ("l", loser, winner)):
            for name, value in self._h2h(tour, a, b, surface).items():
                out[f"{tag}_{name}"] = value
        return out

    # --------------------------------------------------------------- update
    def update(self, row: dict, winner_elo: float, loser_elo: float) -> None:
        """Fold one match result into both players' rolling state."""
        tour, surface, date = row["tour"], row["surface"], row["match_date"]
        winner, loser = int(row["winner_id"]), int(row["loser_id"])
        walkover = bool(row.get("walkover", False))
        retirement = bool(row.get("retirement", False))
        minutes = float(row.get("minutes") or np.nan)
        sets_played = float(row.get("sets_played") or 0)

        for tag, player, opponent_elo, won in (
            ("winner", winner, loser_elo, True),
            ("loser", loser, winner_elo, False),
        ):
            state = self._get(tour, player)
            if not walkover:
                state.results.append((date, 1.0 if won else 0.0, opponent_elo, 1.0))
                state.workload.append((date, minutes, sets_played))
                state.surfaces.append((date, surface))
                state.matches += 1
                state.wins += int(won)
                state.surface_matches[surface] += 1.0
                state.surface_wins[surface] += float(won)
                if won:
                    state.win_streak += 1
                    state.loss_streak = 0
                else:
                    state.loss_streak += 1
                    state.win_streak = 0

                bp_saved = row.get(f"{tag}_bp_saved")
                bp_faced = row.get(f"{tag}_bp_faced")
                if bp_faced is not None and np.isfinite(bp_faced):
                    state.bp_saved += float(bp_saved or 0.0)
                    state.bp_faced += float(bp_faced)
                bp_conv = row.get(f"{tag}_bp_conv")
                bp_opps = row.get(f"{tag}_bp_opps")
                if bp_opps is not None and np.isfinite(bp_opps):
                    state.bp_conv += float(bp_conv or 0.0)
                    state.bp_opps += float(bp_opps)

                tb_played = float(row.get("tiebreaks_played") or 0)
                if tb_played > 0:
                    state.tiebreaks_played += tb_played
                    state.tiebreaks_won += float(row.get(f"{tag}_tiebreaks_won") or 0)

                if bool(row.get("went_to_decider", False)):
                    state.deciders_played += 1
                    state.deciders_won += float(won)

            if retirement and not won:
                # The player who retired is the loser of a retired match.
                state.retirements += 1
                state.recent_retirements.append(date)
            state.last_date = date

        if not walkover:
            self.h2h[(tour, winner, loser)][0] += 1
            self.h2h[(tour, loser, winner)][1] += 1
            self.h2h_surface[(tour, winner, loser, surface)][0] += 1
            self.h2h_surface[(tour, loser, winner, surface)][1] += 1


def run_history(
    matches: pd.DataFrame,
    elo_features: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, HistoryEngine]:
    """Stream matches through the history engine, emitting pre-match features.

    ``elo_features`` supplies the pre-match ratings used to quality-weight form.
    """
    engine = HistoryEngine(config)
    order = matches.sort_values(
        ["match_date", "tour", "tourney_id", "match_num"], kind="mergesort"
    ).index
    records: list[dict[str, float]] = []
    match_rows = matches.loc[order].to_dict("records")
    w_elo = elo_features.loc[order, "w_elo"].to_numpy()
    l_elo = elo_features.loc[order, "l_elo"].to_numpy()

    for position, row in enumerate(match_rows):
        records.append(engine.pre_match_features(row))
        engine.update(row, float(w_elo[position]), float(l_elo[position]))

    frame = pd.DataFrame(records, index=order)
    return frame.reindex(matches.index), engine
