"""Walk-forward backtesting.

The only defensible way to evaluate a sports model. For each evaluation period
the model is trained exclusively on matches played *before* that period, then
scored on it - exactly the information a forecaster would have had at the time.
A cross-validated score on shuffled tennis data is meaningless: ratings, form
and head-to-head records all encode the future once the ordering is broken.

Every run also scores a set of reference models on the identical rows, because
an absolute log loss says nothing on its own. The references are:

``coin``      - always 0.5. The zero-information floor.
``favourite`` - the better-ranked player wins. What a casual observer would say.
``elo``       - the pure blended-Elo logistic. The serious baseline, and the one
                that actually has to be beaten for the ensemble to earn its keep.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CONFIG, ModelConfig
from ..models.ensemble import TennisEnsemble
from . import metrics as M

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    period: str
    tour: str
    n: int
    model: dict
    references: dict = field(default_factory=dict)


def reference_predictions(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Baseline predictions computed directly from the feature matrix."""
    n = len(features)
    out: dict[str, np.ndarray] = {"coin": np.full(n, 0.5)}

    # "Back the better-ranked player" as a probability, via a fixed logistic on
    # the log-rank gap. The scale is the long-run tour value, not fitted here.
    rank_gap = features.get("d_log_rank")
    if rank_gap is not None:
        out["favourite"] = 1.0 / (1.0 + np.exp(-0.62 * rank_gap.fillna(0.0).to_numpy()))

    # Pure Elo, using the standard 400-point logistic. No fitting at all.
    elo_gap = features.get("d_elo_blend")
    if elo_gap is not None:
        out["elo"] = 1.0 / (1.0 + 10.0 ** (-elo_gap.fillna(0.0).to_numpy() / 400.0))
    return out


def walk_forward(
    features: pd.DataFrame,
    start_year: int | None = None,
    min_train_matches: int = 6000,
    per_tour: bool = True,
    config: ModelConfig | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Retrain once per season and score the following season.

    Returns a tidy metrics frame and the per-fold detail.
    """
    features = features.sort_values("match_date", kind="mergesort").reset_index(drop=True)
    features = features[features["match_date"].notna()]
    years = sorted(features["match_date"].dt.year.unique())
    if start_year is None:
        # Start once there is enough history for ratings to have converged.
        cumulative = features.groupby(features["match_date"].dt.year).size().cumsum()
        eligible = [y for y in years if cumulative.get(y, 0) >= min_train_matches]
        start_year = eligible[0] + 1 if eligible else years[len(years) // 2]

    folds: list[FoldResult] = []
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []

    for year in [y for y in years if y >= start_year]:
        train = features[features["match_date"].dt.year < year]
        test = features[features["match_date"].dt.year == year]
        if len(train) < min_train_matches or len(test) < 100:
            continue

        model = TennisEnsemble(config or CONFIG.model).fit(train, verbose=False)
        probability = model.predict(test)
        references = reference_predictions(test)

        block = test[["match_id", "match_date", "tour", "surface", "label"]].copy()
        block["prediction"] = probability
        for name, values in references.items():
            block[f"ref_{name}"] = values
        predictions.append(block)

        groups = [("all", test.index)]
        if per_tour:
            groups += [(tour, test.index[test["tour"] == tour]) for tour in test["tour"].unique()]

        for tour, index in groups:
            if len(index) < 50:
                continue
            position = test.index.get_indexer(index)
            y = test.loc[index, "label"].to_numpy()
            p = probability[position]
            result = M.evaluate(y, p)
            reference_metrics = {
                name: M.evaluate(y, values[position]) for name, values in references.items()
            }
            folds.append(FoldResult(str(year), tour, len(y), result, reference_metrics))
            row = {"year": year, "tour": tour, **result}
            for name, values in reference_metrics.items():
                row[f"{name}_log_loss"] = values["log_loss"]
                row[f"{name}_accuracy"] = values["accuracy"]
            row["skill_vs_elo"] = M.skill_score(y, p, references["elo"][position])
            rows.append(row)

        if verbose:
            overall = [r for r in rows if r["year"] == year and r["tour"] == "all"]
            if overall:
                r = overall[0]
                log.info(
                    "%d: n=%5d logloss=%.4f (elo %.4f) acc=%.4f ece=%.4f slope=%.2f skill=%+.2f%%",
                    year, r["n"], r["log_loss"], r["elo_log_loss"], r["accuracy"],
                    r["ece"], r["calibration_slope"], 100 * r["skill_vs_elo"],
                )

    summary = pd.DataFrame(rows)
    if predictions:
        walk_forward.predictions = pd.concat(predictions, ignore_index=True)
    return summary, folds


def pooled_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics over every out-of-sample prediction, pooled across folds.

    Pooling is the headline number: per-season metrics bounce around because a
    season is only ~2 500 matches, and averaging per-season log losses
    over-weights thin seasons.
    """
    rows = []
    groups = [("all", predictions)]
    for tour in sorted(predictions["tour"].unique()):
        groups.append((tour, predictions[predictions["tour"] == tour]))
    for surface in sorted(predictions["surface"].dropna().unique()):
        groups.append((f"surface:{surface}", predictions[predictions["surface"] == surface]))

    for name, block in groups:
        if len(block) < 100:
            continue
        y = block["label"].to_numpy()
        p = block["prediction"].to_numpy()
        row = {"group": name, **M.evaluate(y, p)}
        for reference in [c for c in block.columns if c.startswith("ref_")]:
            row[f"{reference}_log_loss"] = M.log_loss_safe(y, block[reference].to_numpy())
        row["skill_vs_elo"] = M.skill_score(y, p, block["ref_elo"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)
