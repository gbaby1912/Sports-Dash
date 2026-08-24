"""Assemble the model-ready feature matrix.

Two ideas drive the design.

**Antisymmetry.** A match between A and B has no natural "first" player, and a
model that treats one side as privileged will give inconsistent answers -
P(A beats B) and 1 - P(B beats A) will not agree. Every feature here is either a
*difference* between the two sides or a property of the match itself, and the
side assignment is randomised per match so the label is balanced. That makes the
learned function antisymmetric by construction, and the predictor additionally
averages both orientations at inference time.

**Strict causality.** The rating, serve/return and history engines are each
single chronological passes that read state before writing it, so no feature can
see the match it describes. The only stage that could break this is the periodic
serve/return refit, which is explicitly fitted on ``match_date < cut``.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from ..config import CONFIG, PROCESSED_DIR
from ..data.schema import ROUND_ORDER
from ..data.store import save_frame
from ..data.venues import venue_country
from ..models.markov import match_win_probability
from .elo import run_elo
from .history import run_history
from .serve_return import RollingServeReturn

log = logging.getLogger(__name__)

SURFACE_DUMMIES = ("Hard", "Clay", "Grass", "Carpet")


def _orientation(match_ids: pd.Series) -> np.ndarray:
    """Deterministically decide which side becomes ``p1``.

    Hashing the match id (rather than using a global RNG) means the orientation
    is stable across runs and machines, so a rebuilt feature matrix is
    byte-identical and backtests are reproducible.
    """
    return np.array(
        [
            int(hashlib.blake2b(str(mid).encode(), digest_size=8).hexdigest(), 16) % 2 == 0
            for mid in match_ids
        ]
    )


def _diff(frame: pd.DataFrame, name: str, p1_is_winner: np.ndarray) -> np.ndarray:
    """Signed difference for a feature that exists as ``w_<name>``/``l_<name>``."""
    w = frame[f"w_{name}"].to_numpy(dtype=float)
    l = frame[f"l_{name}"].to_numpy(dtype=float)
    return np.where(p1_is_winner, w - l, l - w)


def _side(frame: pd.DataFrame, name: str, p1_is_winner: np.ndarray, first: bool) -> np.ndarray:
    """Pick out p1's (or p2's) value of a two-sided column."""
    w = frame[f"w_{name}"].to_numpy(dtype=float)
    l = frame[f"l_{name}"].to_numpy(dtype=float)
    take_winner = p1_is_winner if first else ~p1_is_winner
    return np.where(take_winner, w, l)


def _side_object(frame: pd.DataFrame, name: str, p1_is_winner: np.ndarray, first: bool):
    w = frame[f"winner_{name}"].to_numpy()
    l = frame[f"loser_{name}"].to_numpy()
    take_winner = p1_is_winner if first else ~p1_is_winner
    return np.where(take_winner, w, l)


def build_features(
    matches: pd.DataFrame,
    save: bool = True,
    refit_days: int = 28,
) -> tuple[pd.DataFrame, dict]:
    """Build the full feature matrix. Returns (features, fitted engines)."""
    matches = matches.sort_values(
        ["match_date", "tour", "tourney_id", "match_num"], kind="mergesort"
    ).reset_index(drop=True)

    log.info("running Elo engine over %d matches", len(matches))
    elo_features, elo_engine = run_elo(matches)

    log.info("running rolling serve/return fits")
    rolling = RollingServeReturn(refit_days=refit_days)
    sr_features = rolling.build(matches)

    log.info("running rolling history features")
    history_features, history_engine = run_history(matches, elo_features)

    combined = pd.concat([matches, elo_features, sr_features, history_features], axis=1)
    features = assemble(combined)

    if save:
        written = save_frame(features, PROCESSED_DIR / "features")
        log.info("wrote %s (%d rows x %d cols)", written, *features.shape)

    engines = {
        "elo": elo_engine,
        "serve_return": rolling,
        "history": history_engine,
    }
    return features, engines


def assemble(frame: pd.DataFrame, p1_is_winner: np.ndarray | None = None) -> pd.DataFrame:
    """Turn a winner/loser-oriented frame into the antisymmetric p1/p2 matrix.

    ``p1_is_winner`` is randomised per match during training. At prediction time
    there is no winner yet, so the caller pins it to True and the "winner" slot
    simply carries player 1. Reusing this exact function for both keeps training
    and serving on one code path - the usual source of train/serve skew is a
    second, subtly different feature builder written for inference.
    """
    cfg = CONFIG.features
    if p1_is_winner is None:
        p1_is_winner = _orientation(frame["match_id"])
    p1_is_winner = np.asarray(p1_is_winner, dtype=bool)
    out = pd.DataFrame(index=frame.index)

    # --- bookkeeping (never fed to the model) ------------------------------
    out["match_id"] = frame["match_id"].to_numpy()
    out["match_date"] = frame["match_date"].to_numpy()
    out["tour"] = frame["tour"].to_numpy()
    out["surface"] = frame["surface"].to_numpy()
    out["tourney_name"] = frame["tourney_name"].to_numpy()
    out["round"] = frame["round"].to_numpy()
    out["p1_id"] = _side_object(frame, "id", p1_is_winner, True)
    out["p2_id"] = _side_object(frame, "id", p1_is_winner, False)
    out["p1_name"] = _side_object(frame, "name", p1_is_winner, True)
    out["p2_name"] = _side_object(frame, "name", p1_is_winner, False)
    out["label"] = p1_is_winner.astype(int)

    # --- rating differences ------------------------------------------------
    for name in ("elo", "elo_surface", "elo_blend", "elo_points", "elo_games", "elo_peak"):
        out[f"d_{name}"] = _diff(frame, name, p1_is_winner)
    # Rating reliability: the smaller of the two match counts caps how much the
    # rating gap should be trusted, so the model can discount thin samples.
    out["min_elo_matches"] = np.minimum(
        _side(frame, "elo_matches", p1_is_winner, True),
        _side(frame, "elo_matches", p1_is_winner, False),
    )
    out["min_elo_surface_matches"] = np.minimum(
        _side(frame, "elo_surface_matches", p1_is_winner, True),
        _side(frame, "elo_surface_matches", p1_is_winner, False),
    )
    # How far each player sits below their own peak - a decline/rebuild signal.
    out["d_elo_vs_peak"] = _diff(frame, "elo", p1_is_winner) - _diff(frame, "elo_peak", p1_is_winner)

    # --- serve / return ----------------------------------------------------
    for name in ("serve_skill", "return_skill", "serve_skill_surface", "return_skill_surface"):
        out[f"d_{name}"] = _diff(frame, name, p1_is_winner)
    # The cross terms: p1's serve against p2's return is the matchup that
    # actually gets played, and it is not the same as the difference of skills.
    p1_serve = _side(frame, "serve_skill_surface", p1_is_winner, True)
    p2_serve = _side(frame, "serve_skill_surface", p1_is_winner, False)
    p1_return = _side(frame, "return_skill_surface", p1_is_winner, True)
    p2_return = _side(frame, "return_skill_surface", p1_is_winner, False)
    p1_edge = p1_serve - p2_return
    p2_edge = p2_serve - p1_return
    # Stored as an antisymmetric/symmetric pair rather than as two side-specific
    # columns, so that swapping p1 and p2 is an exact sign flip (see
    # `antisymmetric_columns`). Keeping "p1_serve_vs_p2_return" as its own
    # feature would silently break that guarantee.
    out["d_serve_dominance"] = p1_edge - p2_edge
    # Total serve dominance in the match: high means a serve-fest where holds
    # dominate and tiebreaks decide it, which compresses the favourite's edge.
    out["match_serve_level"] = p1_edge + p2_edge
    out["min_sr_points"] = np.minimum(
        _side(frame, "sr_points", p1_is_winner, True),
        _side(frame, "sr_points", p1_is_winner, False),
    )

    # --- the Markov (point-based) view of the matchup ----------------------
    p1_spw = _side(frame, "exp_spw", p1_is_winner, True)
    p2_spw = _side(frame, "exp_spw", p1_is_winner, False)
    best_of = frame["best_of"].to_numpy(dtype=int)
    markov = np.full(len(frame), np.nan)
    valid = np.isfinite(p1_spw) & np.isfinite(p2_spw)
    for bo in (3, 5):
        mask = valid & (best_of == bo)
        if mask.any():
            markov[mask] = match_win_probability(p1_spw[mask], p2_spw[mask], best_of=bo)
    out["d_exp_spw"] = p1_spw - p2_spw
    out["mean_exp_spw"] = 0.5 * (p1_spw + p2_spw)
    out["markov_prob"] = markov
    out["markov_logit"] = _safe_logit(markov)

    # --- form, fatigue, clutch, surface, h2h -------------------------------
    form_names = [f"form_win_pct_{w}" for w in cfg.form_windows] + ["form_decayed", "form_quality"]
    fatigue_names = (
        [f"minutes_{d}d" for d in cfg.fatigue_windows_days]
        + [f"matches_{d}d" for d in cfg.fatigue_windows_days]
        + [f"sets_{d}d" for d in cfg.fatigue_windows_days]
        + ["days_since_last", "rest_deviation", "is_returning", "recent_retirements"]
    )
    clutch_names = ["bp_saved_pct", "bp_conv_pct", "tiebreak_pct", "decider_pct"]
    surface_names = ["surface_win_pct", "surface_matches", "surface_recency"]
    record_names = ["career_matches", "career_win_pct", "win_streak", "loss_streak"]
    # Head-to-head is handled separately below: the match counts are shared by
    # both players and must not be differenced.

    for name in form_names + fatigue_names + clutch_names + surface_names + record_names:
        out[f"d_{name}"] = _diff(frame, name, p1_is_winner)

    out["d_h2h_win_pct"] = _diff(frame, "h2h_win_pct", p1_is_winner)
    out["h2h_matches"] = _side(frame, "h2h_matches", p1_is_winner, True)
    out["d_h2h_surface_win_pct"] = _diff(frame, "h2h_surface_win_pct", p1_is_winner)
    out["h2h_surface_matches"] = _side(frame, "h2h_surface_matches", p1_is_winner, True)

    # --- player attributes -------------------------------------------------
    for name in ("age", "height_cm"):
        w = frame[f"winner_{name}"].to_numpy(dtype=float)
        l = frame[f"loser_{name}"].to_numpy(dtype=float)
        out[f"d_{name}"] = np.where(p1_is_winner, w - l, l - w)
        # The symmetric partner: two 33-year-olds play a different match from
        # two 22-year-olds even when the age *gap* is identical.
        out[f"mean_{name}"] = 0.5 * (w + l)
    # Ageing is not linear: distance from the peak-performance band matters more
    # than raw age, and it is asymmetric (decline is steeper than development).
    w_from_peak = np.abs(frame["winner_age"].to_numpy(dtype=float) - 26.0)
    l_from_peak = np.abs(frame["loser_age"].to_numpy(dtype=float) - 26.0)
    out["d_age_from_peak"] = np.where(p1_is_winner, w_from_peak - l_from_peak,
                                      l_from_peak - w_from_peak)
    out["mean_age_from_peak"] = 0.5 * (w_from_peak + l_from_peak)

    # Ranking enters in log space: the gap between #1 and #10 is far bigger than
    # between #101 and #110, and log rank reflects that.
    for name, transform in (("rank", np.log1p), ("rank_points", np.log1p)):
        w = transform(frame[f"winner_{name}"].to_numpy(dtype=float))
        l = transform(frame[f"loser_{name}"].to_numpy(dtype=float))
        sign = 1.0 if name == "rank_points" else -1.0  # lower rank number is better
        out[f"d_log_{name}"] = sign * np.where(p1_is_winner, w - l, l - w)

    # Handedness matchup: +1 when p1 is a lefty facing a righty, -1 in reverse.
    w_hand = frame["winner_hand"].to_numpy()
    l_hand = frame["loser_hand"].to_numpy()
    p1_hand = np.where(p1_is_winner, w_hand, l_hand)
    p2_hand = np.where(p1_is_winner, l_hand, w_hand)
    out["lefty_edge"] = ((p1_hand == "L") & (p2_hand == "R")).astype(float) - (
        (p1_hand == "R") & (p2_hand == "L")
    ).astype(float)

    # Home advantage: playing in your own country.
    host = frame["tourney_name"].map(venue_country).to_numpy()
    p1_ioc = _side_object(frame, "ioc", p1_is_winner, True)
    p2_ioc = _side_object(frame, "ioc", p1_is_winner, False)
    out["home_edge"] = (p1_ioc == host).astype(float) - (p2_ioc == host).astype(float)

    # --- match context (shared, not differenced) ---------------------------
    for surface in SURFACE_DUMMIES:
        out[f"surface_{surface.lower()}"] = (frame["surface"] == surface).astype(float)
    out["indoor"] = frame["indoor"].astype(float).to_numpy()
    out["altitude_m"] = frame["altitude_m"].astype(float).to_numpy()
    out["best_of"] = best_of.astype(float)
    out["round_order"] = frame["round"].map(ROUND_ORDER).fillna(7).astype(float).to_numpy()
    out["draw_size"] = frame["draw_size"].astype(float).to_numpy()
    out["level_weight"] = frame["level_weight"].astype(float).to_numpy()
    out["is_slam"] = (frame["level"] == "G").astype(float).to_numpy()
    out["tour_is_wta"] = (frame["tour"] == "wta").astype(float).to_numpy()

    return out


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# Features that must flip sign when p1 and p2 are swapped. Everything else in
# the matrix is a property of the match and is invariant under the swap. This
# list is the contract that makes `flip_features` exact - it is asserted in the
# test suite against a real feature matrix.
EXTRA_ANTISYMMETRIC = ("lefty_edge", "home_edge", "markov_logit")


def antisymmetric_columns(columns) -> list[str]:
    """Columns that negate when the two players are swapped."""
    return [
        c for c in columns
        if c.startswith("d_") or c in EXTRA_ANTISYMMETRIC
    ]


def flip_features(features):
    """Return the feature matrix as seen from p2's point of view.

    Used at inference time to average both orientations, which guarantees
    P(A beats B) + P(B beats A) == 1 exactly rather than approximately.
    """
    flipped = features.copy()
    for column in antisymmetric_columns(features.columns):
        flipped[column] = -flipped[column]
    if "markov_prob" in flipped.columns:
        flipped["markov_prob"] = 1.0 - flipped["markov_prob"]
    return flipped


# Columns that identify a row rather than describe it. Excluded from training.
META_COLUMNS = [
    "match_id", "match_date", "tour", "surface", "tourney_name", "round",
    "p1_id", "p2_id", "p1_name", "p2_name", "label",
]


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model input columns, in a stable order."""
    return [c for c in features.columns if c not in META_COLUMNS]
