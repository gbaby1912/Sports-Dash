"""Stacked, calibrated ensemble.

Training protocol, and why it is shaped this way:

1. Split the training window **chronologically** into a fit block and a later
   holdout block. Random splits are wrong here - tennis data is a time series,
   and a random split lets the model learn from the future.
2. Fit the base learners on the fit block.
3. Predict the holdout with those learners and fit the stacker on *those*
   out-of-sample predictions. Fitting the stacker on in-sample base predictions
   is the classic stacking mistake: it hands all the weight to whichever member
   overfits hardest.
4. Fit the probability calibrator on the same holdout.
5. Refit the base learners on the full training window so the shipped model uses
   every available match, keeping the stacker weights and calibrator learned in
   steps 3-4.

At prediction time the matrix is scored in both orientations and averaged, which
makes ``P(A beats B) + P(B beats A) == 1`` hold exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..config import CONFIG, ModelConfig
from ..features.builder import feature_columns, flip_features
from .base_learners import BaseLearner, LearnerSet, _logit

log = logging.getLogger(__name__)


@dataclass
class Calibrator:
    """Maps raw model probabilities onto observed frequencies."""

    method: str = "isotonic"
    _isotonic: IsotonicRegression | None = None
    _platt: LogisticRegression | None = None
    fitted: bool = False

    def fit(self, probabilities: np.ndarray, y: np.ndarray, min_isotonic: int = 4000) -> "Calibrator":
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic" and len(y) >= min_isotonic:
            # out_of_bounds="clip" keeps predictions finite outside the observed
            # range, which matters for the rare extreme mismatch.
            self._isotonic = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._isotonic.fit(probabilities, y)
        else:
            self._platt = LogisticRegression(C=1e6, max_iter=1000)
            self._platt.fit(_logit(probabilities).reshape(-1, 1), y)
            self.method = "platt"
        self.fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if not self.fitted:
            return probabilities
        if self._isotonic is not None:
            return np.clip(self._isotonic.predict(probabilities), 0.01, 0.99)
        return self._platt.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


@dataclass
class TennisEnsemble:
    """The full stack: base learners -> logistic stacker -> calibrator."""

    config: ModelConfig = field(default_factory=lambda: CONFIG.model)
    learners: list[BaseLearner] = field(default_factory=list)
    stacker: LogisticRegression | None = None
    calibrator: Calibrator | None = None
    columns: list[str] = field(default_factory=list)
    training_rows: int = 0
    trained_through: pd.Timestamp | None = None
    holdout_metrics: dict = field(default_factory=dict)
    feature_bounds: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- train
    def fit(self, features: pd.DataFrame, verbose: bool = True) -> "TennisEnsemble":
        cfg = self.config
        features = features.sort_values("match_date", kind="mergesort")
        self.columns = feature_columns(features)
        X = features[self.columns]
        y = features["label"].to_numpy()

        split = int(len(features) * (1.0 - cfg.holdout_fraction))
        split = max(split, 1)
        X_fit, y_fit = X.iloc[:split], y[:split]
        X_hold, y_hold = X.iloc[split:], y[split:]
        if len(X_hold) < 200:
            # Too little data to stack honestly; fall back to a single learner.
            log.warning("holdout of %d rows is too small for stacking", len(X_hold))
            X_fit, y_fit = X, y
            X_hold, y_hold = X, y

        # --- 1-2: base learners on the fit block ---------------------------
        self.learners = LearnerSet.build(cfg)
        for learner in self.learners:
            learner.fit(X_fit, y_fit)
            if verbose:
                from ..backtest.metrics import log_loss_safe
                probability = learner.predict_proba(X_hold)
                log.info(
                    "  base %-8s holdout logloss=%.4f acc=%.4f",
                    learner.name,
                    log_loss_safe(y_hold, probability),
                    ((probability > 0.5) == (y_hold == 1)).mean(),
                )

        # --- 3: stacker on out-of-sample base predictions ------------------
        meta_hold = self._meta_matrix(X_hold)
        self.stacker = LogisticRegression(C=1.0, max_iter=2000)
        self.stacker.fit(meta_hold, y_hold)

        # --- 4: calibration on the same holdout ----------------------------
        raw = self.stacker.predict_proba(meta_hold)[:, 1]
        self.calibrator = Calibrator(cfg.calibration).fit(
            raw, y_hold, min_isotonic=cfg.isotonic_min_samples
        )

        from ..backtest.metrics import evaluate
        self.holdout_metrics = evaluate(y_hold, self.calibrator.transform(raw))

        # --- 5: refit base learners on everything --------------------------
        if len(X_hold) != len(X):
            for learner in self.learners:
                learner.fit(X, y)

        self.feature_bounds = self._compute_bounds(X)
        self.training_rows = len(features)
        self.trained_through = features["match_date"].max()
        return self

    @staticmethod
    def _compute_bounds(X: pd.DataFrame) -> dict[str, tuple[float, float]]:
        """The 1st-99th percentile envelope of each feature in training.

        Used to clamp inputs at prediction time. A hypothetical matchup is not
        drawn from the same distribution as a real scheduled match - two players
        can be picked who last played a year apart, which essentially never
        happens inside a live draw - and the linear members extrapolate happily
        and badly outside the range they were fitted on.

        Antisymmetric features get a *symmetric* bound, +/- the larger tail.
        Clipping them to an asymmetric range would silently break the exact
        antisymmetry guarantee, because clip(-x) would stop equalling -clip(x).
        """
        from ..features.builder import antisymmetric_columns

        antisymmetric = set(antisymmetric_columns(X.columns))
        bounds: dict[str, tuple[float, float]] = {}
        low = X.quantile(0.01)
        high = X.quantile(0.99)
        for column in X.columns:
            lo, hi = float(low[column]), float(high[column])
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                continue
            if column in antisymmetric:
                limit = max(abs(lo), abs(hi))
                bounds[column] = (-limit, limit)
            else:
                bounds[column] = (lo, hi)
        return bounds

    def clamp(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clip a feature frame into the envelope the model was trained on."""
        if not self.feature_bounds:
            return features
        clamped = features.copy()
        for column, (lo, hi) in self.feature_bounds.items():
            if column in clamped.columns:
                clamped[column] = clamped[column].clip(lo, hi)
        return clamped

    def _meta_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Stacker inputs: each base learner's logit, side by side."""
        return np.column_stack([learner.predict_logit(X) for learner in self.learners])

    # --------------------------------------------------------------- predict
    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated probability for the matrix exactly as given (no flip)."""
        X = features[self.columns]
        raw = self.stacker.predict_proba(self._meta_matrix(X))[:, 1]
        return self.calibrator.transform(raw)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Symmetry-averaged probability that p1 wins.

        Scoring both orientations and averaging costs one extra forward pass and
        buys an exact guarantee: the model can never claim both players are
        favourites.
        """
        forward = self.predict_raw(features)
        backward = self.predict_raw(flip_features(features))
        return np.clip(0.5 * (forward + (1.0 - backward)), 0.001, 0.999)

    def base_predictions(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        """Each member's probability, for the dashboard's model breakdown."""
        X = features[self.columns]
        return {learner.name: learner.predict_proba(X) for learner in self.learners}

    @property
    def stacker_weights(self) -> dict[str, float]:
        """How much the stacker leans on each member, for transparency."""
        if self.stacker is None:
            return {}
        return {
            learner.name: float(coefficient)
            for learner, coefficient in zip(self.learners, self.stacker.coef_[0])
        }

    def feature_importance(self, features: pd.DataFrame, n_repeats: int = 3,
                           sample: int = 6000, seed: int = 0) -> pd.DataFrame:
        """Permutation importance of each feature on the ensemble output.

        Measured on the ensemble rather than on the GBM alone, so it reflects
        what actually drives the shipped predictions.
        """
        from sklearn.metrics import log_loss

        rng = np.random.default_rng(seed)
        frame = features.sample(min(sample, len(features)), random_state=seed)
        y = frame["label"].to_numpy()
        baseline = log_loss(y, self.predict(frame), labels=[0, 1])

        rows = []
        for column in self.columns:
            deltas = []
            for _ in range(n_repeats):
                shuffled = frame.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                deltas.append(log_loss(y, self.predict(shuffled), labels=[0, 1]) - baseline)
            rows.append({"feature": column, "importance": float(np.mean(deltas))})
        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
