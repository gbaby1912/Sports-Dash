"""Opponent-adjusted serve and return skill.

Raw serve-points-won is one of the most-quoted tennis stats and one of the most
misleading. It is schedule-contaminated: a player who spent the season drawing
elite returners looks worse than they are, and a player who farmed weak fields
looks better. The same is true in reverse for return stats.

The fix is to stop treating a player's serve numbers as a property of that
player and start treating each *match* as an observation of a difference:

    logit(spw_ij) = mu[tour, surface] + serve_i + serve_i@surface
                                      - return_j - return_j@surface

Fitting that additive model by ridge-penalised weighted least squares recovers
serve and return skill on a common scale, purged of who each player happened to
face. This is the same idea as adjusted plus-minus in basketball.

Three design choices matter:

* **Logit space.** Serve percentages live in a narrow band and their variance
  is not constant across that band; the logit makes the additive form sensible
  and keeps predictions inside (0, 1).
* **Weighting by service points, with exponential time decay.** A match with
  180 service points is worth more than one with 50, and a match from three
  years ago is worth less than one from last month.
* **Ridge shrinkage toward tour average.** A player with 200 career service
  points should not be credited with an extreme rating. The penalty does this
  automatically and continuously, with no arbitrary cutoff.

The fitted skills feed the Markov model, which turns them into a match win
probability - a completely different route to a prediction than Elo takes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CONFIG, ServeReturnConfig

log = logging.getLogger(__name__)

_EPS = 1e-6


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.02, 0.98)
    return np.log(x / (1.0 - x))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class ServeReturnFit:
    """A fitted snapshot of serve/return skill at one point in time."""

    as_of: pd.Timestamp
    serve: dict[tuple[str, int], float] = field(default_factory=dict)
    ret: dict[tuple[str, int], float] = field(default_factory=dict)
    serve_surface: dict[tuple[str, int, str], float] = field(default_factory=dict)
    ret_surface: dict[tuple[str, int, str], float] = field(default_factory=dict)
    baseline: dict[tuple[str, str], float] = field(default_factory=dict)
    serve_points: dict[tuple[str, int], float] = field(default_factory=dict)
    return_points: dict[tuple[str, int], float] = field(default_factory=dict)
    # Raw (unadjusted) rates, kept for display so the dashboard can show both.
    raw_spw: dict[tuple[str, int], float] = field(default_factory=dict)
    raw_rpw: dict[tuple[str, int], float] = field(default_factory=dict)

    def serve_skill(self, tour: str, player: int, surface: str | None = None) -> float:
        base = self.serve.get((tour, player), 0.0)
        if surface:
            base += self.serve_surface.get((tour, player, surface), 0.0)
        return base

    def return_skill(self, tour: str, player: int, surface: str | None = None) -> float:
        base = self.ret.get((tour, player), 0.0)
        if surface:
            base += self.ret_surface.get((tour, player, surface), 0.0)
        return base

    def tour_baseline(self, tour: str, surface: str) -> float:
        if (tour, surface) in self.baseline:
            return self.baseline[(tour, surface)]
        # Fall back to the tour's average across surfaces.
        values = [v for (t, _), v in self.baseline.items() if t == tour]
        return float(np.mean(values)) if values else _logit(np.array([0.62]))[0]

    def spw_vs_average_returner(self, tour: str, server: int, surface: str) -> float:
        """Serve percentage this player would post against a league-average returner.

        The neutral, schedule-free version of "serve points won" - the number a
        raw career percentage is trying and failing to be.
        """
        value = self.tour_baseline(tour, surface) + self.serve_skill(tour, server, surface)
        return float(np.clip(_expit(np.array([value]))[0], 0.30, 0.90))

    def rpw_vs_average_server(self, tour: str, returner: int, surface: str) -> float:
        """Return points won against a league-average server."""
        value = self.tour_baseline(tour, surface) - self.return_skill(tour, returner, surface)
        return float(1.0 - np.clip(_expit(np.array([value]))[0], 0.30, 0.90))

    def expected_spw(self, tour: str, server: int, returner: int, surface: str) -> float:
        """Probability the server wins a point against this specific returner."""
        value = (
            self.tour_baseline(tour, surface)
            + self.serve_skill(tour, server, surface)
            - self.return_skill(tour, returner, surface)
        )
        return float(np.clip(_expit(np.array([value]))[0], 0.30, 0.90))

    def coverage(self, tour: str, player: int) -> float:
        """Service points behind this player's rating - a confidence proxy."""
        return self.serve_points.get((tour, player), 0.0)

    def has_rating(self, tour: str, player: int) -> bool:
        """Whether this player appears in the fit at all.

        `serve_skill` returns 0.0 for an unknown player, which is the right
        default for arithmetic but wrong for display: it reads as "exactly tour
        average" when the truth is "no data in the window". Callers that render
        a number to a human should check this first and show a blank instead.
        """
        return (tour, player) in self.serve


def _build_observations(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: each player's service performance against the other."""
    usable = matches[matches["has_serve_stats"].fillna(False)]
    usable = usable[~usable["walkover"].fillna(False)]
    if usable.empty:
        return pd.DataFrame()

    frames = []
    for server, returner in (("winner", "loser"), ("loser", "winner")):
        block = pd.DataFrame(
            {
                "tour": usable["tour"].to_numpy(),
                "surface": usable["surface"].to_numpy(),
                "date": usable["match_date"].to_numpy(),
                "server": usable[f"{server}_id"].to_numpy(),
                "returner": usable[f"{returner}_id"].to_numpy(),
                "svpt": usable[f"{server}_svpt"].to_numpy(dtype=float),
                "spw": usable[f"{server}_spw"].to_numpy(dtype=float),
            }
        )
        frames.append(block)

    observations = pd.concat(frames, ignore_index=True)
    observations = observations[observations["svpt"] > 0]
    observations["rate"] = observations["spw"] / observations["svpt"]
    return observations


def fit_serve_return(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    config: ServeReturnConfig | None = None,
) -> ServeReturnFit:
    """Fit serve/return skill using only matches strictly before ``as_of``."""
    cfg = config or CONFIG.serve_return
    history = matches[matches["match_date"] < as_of]
    window_start = as_of - pd.Timedelta(days=cfg.window_days)
    history = history[history["match_date"] >= window_start]

    observations = _build_observations(history)
    fit = ServeReturnFit(as_of=as_of)
    if observations.empty:
        return fit

    # --- weights: service points, exponentially decayed by recency ---------
    age_days = (as_of - pd.to_datetime(observations["date"])).dt.days.to_numpy(dtype=float)
    decay = 0.5 ** (age_days / cfg.half_life_days)
    weight = observations["svpt"].to_numpy(dtype=float) * decay
    y = _logit(observations["rate"].to_numpy(dtype=float))

    # --- index encoding ----------------------------------------------------
    server_keys = list(zip(observations["tour"], observations["server"]))
    returner_keys = list(zip(observations["tour"], observations["returner"]))
    player_index: dict[tuple[str, int], int] = {}
    for key in server_keys + returner_keys:
        player_index.setdefault(key, len(player_index))
    n_players = len(player_index)

    server_idx = np.fromiter((player_index[k] for k in server_keys), dtype=np.int64, count=len(y))
    returner_idx = np.fromiter((player_index[k] for k in returner_keys), dtype=np.int64, count=len(y))

    group_keys = list(zip(observations["tour"], observations["surface"]))
    group_index: dict[tuple[str, str], int] = {}
    for key in group_keys:
        group_index.setdefault(key, len(group_index))
    group_idx = np.fromiter((group_index[k] for k in group_keys), dtype=np.int64, count=len(y))
    n_groups = len(group_index)

    surfaces = sorted({s for _, s in group_keys})
    surface_index = {s: i for i, s in enumerate(surfaces)}
    surf_idx = np.fromiter(
        (surface_index[s] for s in observations["surface"]), dtype=np.int64, count=len(y)
    )
    # Per-player-per-surface slots, allocated only for pairs that occur.
    serve_surf_key = server_idx * len(surfaces) + surf_idx
    ret_surf_key = returner_idx * len(surfaces) + surf_idx
    n_surface_slots = n_players * len(surfaces)

    # --- coordinate descent on the ridge objective -------------------------
    serve = np.zeros(n_players)
    ret = np.zeros(n_players)
    serve_surf = np.zeros(n_surface_slots)
    ret_surf = np.zeros(n_surface_slots)
    mu = np.zeros(n_groups)

    w_serve = np.bincount(server_idx, weights=weight, minlength=n_players)
    w_ret = np.bincount(returner_idx, weights=weight, minlength=n_players)
    w_group = np.bincount(group_idx, weights=weight, minlength=n_groups)
    w_serve_surf = np.bincount(serve_surf_key, weights=weight, minlength=n_surface_slots)
    w_ret_surf = np.bincount(ret_surf_key, weights=weight, minlength=n_surface_slots)

    lam, lam_s = cfg.ridge_lambda, cfg.surface_ridge_lambda
    previous = None
    for iteration in range(cfg.max_iter):
        # Baseline (unpenalised): weighted mean of the unexplained part.
        partial = y - serve[server_idx] - serve_surf[serve_surf_key] \
                    + ret[returner_idx] + ret_surf[ret_surf_key]
        mu = np.bincount(group_idx, weights=weight * partial, minlength=n_groups) / np.maximum(
            w_group, _EPS
        )

        # Serve skill.
        partial = y - mu[group_idx] - serve_surf[serve_surf_key] \
                    + ret[returner_idx] + ret_surf[ret_surf_key]
        serve = np.bincount(server_idx, weights=weight * partial, minlength=n_players) / (
            w_serve + lam
        )

        # Return skill (enters with a negative sign, hence the flip).
        partial = mu[group_idx] + serve[server_idx] + serve_surf[serve_surf_key] \
                    - ret_surf[ret_surf_key] - y
        ret = np.bincount(returner_idx, weights=weight * partial, minlength=n_players) / (
            w_ret + lam
        )

        # Per-surface deviations, penalised harder.
        partial = y - mu[group_idx] - serve[server_idx] \
                    + ret[returner_idx] + ret_surf[ret_surf_key]
        serve_surf = np.bincount(
            serve_surf_key, weights=weight * partial, minlength=n_surface_slots
        ) / (w_serve_surf + lam_s)

        partial = mu[group_idx] + serve[server_idx] + serve_surf[serve_surf_key] \
                    - ret[returner_idx] - y
        ret_surf = np.bincount(
            ret_surf_key, weights=weight * partial, minlength=n_surface_slots
        ) / (w_ret_surf + lam_s)

        current = np.concatenate([serve, ret, serve_surf, ret_surf, mu])
        if previous is not None and np.max(np.abs(current - previous)) < cfg.tol:
            break
        previous = current

    # NOTE: no post-hoc shrinkage is applied here. The ridge penalty already
    # shrinks each player by exactly w/(w + lambda), and applying a second,
    # independent shrink to serve and return separately would break the
    # identity  mu + serve - return = fitted logit, biasing expected_spw.
    # Thin samples are handled by choosing lambda in service-point units.

    # --- pack the result ---------------------------------------------------
    reverse = {v: k for k, v in player_index.items()}
    for index, key in reverse.items():
        fit.serve[key] = float(serve[index])
        fit.ret[key] = float(ret[index])
        fit.serve_points[key] = float(w_serve[index])
        fit.return_points[key] = float(w_ret[index])
        for surface, s_i in surface_index.items():
            slot = index * len(surfaces) + s_i
            if w_serve_surf[slot] > 0:
                fit.serve_surface[(key[0], key[1], surface)] = float(serve_surf[slot])
            if w_ret_surf[slot] > 0:
                fit.ret_surface[(key[0], key[1], surface)] = float(ret_surf[slot])
    for key, index in group_index.items():
        fit.baseline[key] = float(mu[index])

    # Raw rates for display alongside the adjusted numbers.
    raw_serve = observations.groupby(server_keys if False else [
        observations["tour"], observations["server"]
    ]).apply(lambda g: (g["spw"] * 1.0).sum() / max(g["svpt"].sum(), 1), include_groups=False)
    for key, value in raw_serve.items():
        fit.raw_spw[(key[0], int(key[1]))] = float(value)
    raw_ret = observations.groupby([observations["tour"], observations["returner"]]).apply(
        lambda g: 1.0 - (g["spw"].sum() / max(g["svpt"].sum(), 1)), include_groups=False
    )
    for key, value in raw_ret.items():
        fit.raw_rpw[(key[0], int(key[1]))] = float(value)

    return fit


class RollingServeReturn:
    """Periodically-refit serve/return skill, evaluated leak-free.

    Refitting before every match would be prohibitively slow and would add
    nothing: skill estimates barely move day to day. Instead we refit on a fixed
    cadence using only prior data, and every match in the following window is
    scored with the fit that was available *before* it was played.
    """

    def __init__(self, config: ServeReturnConfig | None = None, refit_days: int = 28) -> None:
        self.config = config or CONFIG.serve_return
        self.refit_days = refit_days
        self.fits: list[ServeReturnFit] = []

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return per-match pre-match serve/return features for every row."""
        ordered = matches.sort_values("match_date", kind="mergesort")
        start = ordered["match_date"].min()
        end = ordered["match_date"].max()
        if pd.isna(start):
            return pd.DataFrame(index=matches.index)

        # Cut points: the model in force changes on this cadence.
        cut_dates = pd.date_range(start, end + pd.Timedelta(days=self.refit_days),
                                  freq=f"{self.refit_days}D")
        records: list[dict] = []
        index_order: list = []
        current: ServeReturnFit | None = None

        for i, cut in enumerate(cut_dates):
            next_cut = cut_dates[i + 1] if i + 1 < len(cut_dates) else end + pd.Timedelta(days=1)
            window = ordered[(ordered["match_date"] >= cut) & (ordered["match_date"] < next_cut)]
            if window.empty:
                continue
            current = fit_serve_return(ordered, as_of=cut, config=self.config)
            self.fits.append(current)
            for row in window.to_dict("records"):
                records.append(_row_features(current, row))
            index_order.extend(window.index.tolist())

        if not records:
            return pd.DataFrame(index=matches.index)
        frame = pd.DataFrame(records, index=index_order)
        return frame.reindex(matches.index)

    @property
    def latest(self) -> ServeReturnFit | None:
        return self.fits[-1] if self.fits else None


def _row_features(fit: ServeReturnFit, row: dict) -> dict[str, float]:
    tour, surface = row["tour"], row["surface"]
    winner, loser = int(row["winner_id"]), int(row["loser_id"])
    out: dict[str, float] = {}
    for tag, player in (("w", winner), ("l", loser)):
        out[f"{tag}_serve_skill"] = fit.serve_skill(tour, player)
        out[f"{tag}_return_skill"] = fit.return_skill(tour, player)
        out[f"{tag}_serve_skill_surface"] = fit.serve_skill(tour, player, surface)
        out[f"{tag}_return_skill_surface"] = fit.return_skill(tour, player, surface)
        out[f"{tag}_sr_points"] = fit.coverage(tour, player)
        out[f"{tag}_raw_spw"] = fit.raw_spw.get((tour, player), np.nan)
        out[f"{tag}_raw_rpw"] = fit.raw_rpw.get((tour, player), np.nan)
    # The matchup-specific serve percentages that feed the Markov model.
    out["w_exp_spw"] = fit.expected_spw(tour, winner, loser, surface)
    out["l_exp_spw"] = fit.expected_spw(tour, loser, winner, surface)
    return out
