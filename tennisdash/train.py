"""Train the shipped model and write the deployable bundle.

The bundle is everything the API needs and nothing it does not: the calibrated
ensemble, the three fitted state engines, the player directory, and a model card
recording how the thing was built and how well it scored. Shipping the metrics
inside the artifact means the dashboard can always show the model's real
measured accuracy rather than a number someone typed into a template.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACT_DIR, CONFIG, PROCESSED_DIR
from .data.ingest import load_match_table
from .data.store import frame_exists, load_frame
from .features.builder import build_features
from .models.ensemble import TennisEnsemble
from .players import build_directory

log = logging.getLogger(__name__)

BUNDLE_PATH = ARTIFACT_DIR / "model.joblib"


def load_or_build_features(rebuild: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load the cached feature matrix, or build it (and the engines) fresh.

    The engines are stateful and cannot be recovered from the saved matrix, so a
    cached matrix alone is not enough to serve predictions - a rebuild is
    required whenever the engines are missing.
    """
    engines_path = ARTIFACT_DIR / "engines.joblib"
    if not rebuild and frame_exists(PROCESSED_DIR / "features") and engines_path.exists():
        log.info("loading cached feature matrix and engines")
        return load_frame(PROCESSED_DIR / "features"), joblib.load(engines_path)

    matches = load_match_table()
    features, engines = build_features(matches)
    joblib.dump(engines, engines_path)
    return features, engines


def train(
    rebuild_features: bool = False,
    backtest: bool = True,
    importance: bool = True,
) -> dict:
    """Build features, run the backtest, fit the final model, save the bundle."""
    started = time.time()
    features, engines = load_or_build_features(rebuild=rebuild_features)
    matches = load_match_table()

    report: dict = {}
    if backtest:
        from .backtest.walkforward import pooled_summary, walk_forward

        log.info("running walk-forward backtest")
        summary, _ = walk_forward(features)
        pooled = pooled_summary(walk_forward.predictions)
        report["backtest_by_season"] = summary.to_dict("records")
        report["backtest_pooled"] = pooled.to_dict("records")
        from .backtest.metrics import calibration_table

        predictions = walk_forward.predictions
        report["calibration"] = calibration_table(
            predictions["label"].to_numpy(), predictions["prediction"].to_numpy()
        ).to_dict("records")

    log.info("fitting final model on all %d matches", len(features))
    ensemble = TennisEnsemble(CONFIG.model).fit(features)

    if importance:
        log.info("computing permutation importance")
        recent = features[features["match_date"] >= features["match_date"].max()
                          - pd.Timedelta(days=730)]
        report["feature_importance"] = (
            ensemble.feature_importance(recent).head(30).to_dict("records")
        )

    directory = build_directory(matches)
    bundle = {
        "ensemble": ensemble,
        "engines": engines,
        "directory": directory,
        "metadata": {
            "trained_at": pd.Timestamp.now().isoformat(),
            "training_rows": int(len(features)),
            "trained_through": str(ensemble.trained_through),
            "data_span": [str(matches["match_date"].min()), str(matches["match_date"].max())],
            "tours": sorted(matches["tour"].unique().tolist()),
            "n_players": int(len(directory)),
            "n_features": len(ensemble.columns),
            "features": list(ensemble.columns),
            "stacker_weights": ensemble.stacker_weights,
            "calibration_method": ensemble.calibrator.method if ensemble.calibrator else None,
            "holdout_metrics": ensemble.holdout_metrics,
            "config": CONFIG.to_dict(),
            "report": report,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    log.info("wrote %s in %.1fs", BUNDLE_PATH, time.time() - started)
    return bundle


def load_bundle(path: Path | None = None) -> dict:
    path = Path(path or BUNDLE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `tennisdash train` (or `make train`)."
        )
    return joblib.load(path)
