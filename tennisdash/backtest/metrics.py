"""Scoring rules and calibration diagnostics.

Accuracy is the least useful number in the list and the one most often quoted.
A tennis model that predicts the higher-ranked player every time already gets
about 65% right, and a model can improve accuracy while getting *worse* at the
thing that matters - saying how confident it is.

Log loss and the Brier score are proper scoring rules: they are minimised only
by reporting your true belief, so they punish both bad discrimination and bad
confidence. Calibration error is reported separately because a model can
discriminate well and still be systematically over-confident, which is the
failure mode that ruins a model in use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-15


def log_loss_safe(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) == 1)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 12) -> float:
    """Average |predicted - observed| across equal-count probability bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) < bins * 5:
        bins = max(2, len(p) // 20)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total, error = 0.0, 0.0
    for i in range(bins):
        mask = (p > edges[i]) & (p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        error += mask.sum() * abs(p[mask].mean() - y[mask].mean())
        total += mask.sum()
    return float(error / total) if total else float("nan")


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed win rate per bin - the calibration plot's data."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for i in range(bins):
        mask = (p > edges[i]) & (p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of observed log-odds on predicted log-odds.

    1.0 is perfect. Below 1 means over-confident (the usual failure); above 1
    means under-confident. This is a sharper diagnostic than ECE because it says
    which *direction* the miscalibration runs.
    """
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logits, y)
    return float(model.coef_[0][0])


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """The full metric set for one block of predictions."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "log_loss": log_loss_safe(y, p),
        "brier": brier(y, p),
        "accuracy": accuracy(y, p),
        "ece": expected_calibration_error(y, p),
        "calibration_slope": calibration_slope(y, p),
        "mean_prediction": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
    }


def skill_score(y: np.ndarray, p: np.ndarray, reference: np.ndarray) -> float:
    """Fractional log-loss improvement over a reference model.

    Positive means better than the reference. This is the honest way to report
    "how good is this model" - raw log loss is meaningless without knowing how
    hard the sample was.
    """
    model_loss = log_loss_safe(y, p)
    reference_loss = log_loss_safe(y, reference)
    if reference_loss <= 0:
        return float("nan")
    return float((reference_loss - model_loss) / reference_loss)


def betting_return(
    y: np.ndarray,
    p: np.ndarray,
    odds_p1: np.ndarray,
    odds_p2: np.ndarray,
    edge_threshold: float = 0.04,
    stake: str = "kelly",
    kelly_fraction: float = 0.25,
) -> dict[str, float]:
    """Flat or fractional-Kelly return against posted decimal odds.

    This is the only external yardstick that cannot be gamed by choosing a
    convenient metric: the closing line is a well-informed forecast, so beating
    it is meaningful and failing to is informative. It is reported as a
    diagnostic, not as a recommendation to bet.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    odds_p1 = np.asarray(odds_p1, dtype=float)
    odds_p2 = np.asarray(odds_p2, dtype=float)

    valid = np.isfinite(odds_p1) & np.isfinite(odds_p2) & (odds_p1 > 1) & (odds_p2 > 1)
    if not valid.any():
        return {"bets": 0, "roi": float("nan"), "profit": float("nan")}

    y, p, odds_p1, odds_p2 = y[valid], p[valid], odds_p1[valid], odds_p2[valid]
    edge_p1 = p * odds_p1 - 1.0
    edge_p2 = (1 - p) * odds_p2 - 1.0

    bet_p1 = edge_p1 > edge_threshold
    bet_p2 = (edge_p2 > edge_threshold) & ~bet_p1

    def kelly(prob, odds):
        b = odds - 1.0
        return np.clip((prob * b - (1 - prob)) / np.maximum(b, 1e-9), 0, 1) * kelly_fraction

    stakes = np.zeros(len(y))
    returns = np.zeros(len(y))
    if stake == "kelly":
        stakes[bet_p1] = kelly(p[bet_p1], odds_p1[bet_p1])
        stakes[bet_p2] = kelly(1 - p[bet_p2], odds_p2[bet_p2])
    else:
        stakes[bet_p1 | bet_p2] = 1.0

    returns[bet_p1] = np.where(y[bet_p1] == 1, stakes[bet_p1] * (odds_p1[bet_p1] - 1), -stakes[bet_p1])
    returns[bet_p2] = np.where(y[bet_p2] == 0, stakes[bet_p2] * (odds_p2[bet_p2] - 1), -stakes[bet_p2])

    staked = stakes.sum()
    return {
        "bets": int((stakes > 0).sum()),
        "staked": float(staked),
        "profit": float(returns.sum()),
        "roi": float(returns.sum() / staked) if staked > 0 else float("nan"),
    }
