"""The individual models that the ensemble stacks.

Each one is a genuinely different view of the same match, which is the point:
stacking only helps when the members make *different* mistakes.

``RatingLearner``
    A small logistic regression over the Elo block. Fast-moving, robust, and the
    hardest baseline to beat in tennis.
``MarkovLearner``
    Recalibrates the point-based Markov probability. The Markov model has no
    free parameters at all, so it needs a one-dimensional recalibration to sit
    on the same scale as the others - its raw output is systematically
    over-confident because it assumes points are independent, which understates
    the variance of a real match.
``GradientBoostingLearner``
    Gradient-boosted trees over the full feature matrix. Captures the
    interactions - surface x style, fatigue x best-of-five, rating gap x
    reliability - that the linear members cannot.
``LinearLearner``
    L2 logistic regression over the full (standardised) matrix. Adds stability
    where the tree model is data-hungry, and degrades gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CONFIG, ModelConfig

RATING_FEATURES = [
    "d_elo", "d_elo_surface", "d_elo_blend", "d_elo_points", "d_elo_games",
    "d_elo_peak", "d_elo_vs_peak", "min_elo_matches", "min_elo_surface_matches",
    "best_of", "tour_is_wta",
]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class BaseLearner:
    """Common interface: fit on a feature frame, emit a probability."""

    name = "base"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseLearner":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        return _logit(self.predict_proba(X))


class RatingLearner(BaseLearner):
    name = "rating"

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        )
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RatingLearner":
        self.columns = [c for c in RATING_FEATURES if c in X.columns]
        self.pipeline.fit(X[self.columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.columns])[:, 1]


class MarkovLearner(BaseLearner):
    """Recalibrates the parameter-free point-based model.

    The Markov model assumes points are i.i.d. within a match. Real matches have
    within-match momentum and correlated errors, so its probabilities are
    systematically too extreme. A single logistic recalibration on the Markov
    logit fixes almost all of that, and keeps the model's independent signal.
    """

    name = "markov"

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(C=1.0, max_iter=1000)),
            ]
        )
        self.columns = ["markov_logit", "match_serve_level", "best_of"]
        self.available = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MarkovLearner":
        self.columns = [c for c in self.columns if c in X.columns]
        usable = X[self.columns]
        self.available = bool(self.columns) and usable["markov_logit"].notna().mean() > 0.2
        if self.available:
            self.pipeline.fit(usable, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.available:
            return np.full(len(X), 0.5)
        return self.pipeline.predict_proba(X[self.columns])[:, 1]


class GradientBoostingLearner(BaseLearner):
    name = "gbm"

    def __init__(self, config: ModelConfig | None = None) -> None:
        cfg = config or CONFIG.model
        self.config = cfg
        self.model = HistGradientBoostingClassifier(
            max_iter=cfg.gbm_max_iter,
            learning_rate=cfg.gbm_learning_rate,
            max_leaf_nodes=cfg.gbm_max_leaf_nodes,
            min_samples_leaf=cfg.gbm_min_samples_leaf,
            l2_regularization=cfg.gbm_l2_regularization,
            max_depth=cfg.gbm_max_depth,
            early_stopping=True,
            n_iter_no_change=cfg.gbm_early_stopping_rounds,
            validation_fraction=0.12,
            random_state=cfg.random_state,
        )
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GradientBoostingLearner":
        self.columns = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.columns])[:, 1]


class LinearLearner(BaseLearner):
    name = "linear"

    def __init__(self, C: float = 0.35) -> None:
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=C, max_iter=3000)),
            ]
        )
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearLearner":
        self.columns = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.columns])[:, 1]


@dataclass
class LearnerSet:
    """The default roster of base learners."""

    @staticmethod
    def build(config: ModelConfig | None = None) -> list[BaseLearner]:
        return [
            RatingLearner(),
            MarkovLearner(),
            GradientBoostingLearner(config),
            LinearLearner(),
        ]
