"""Central configuration: paths, surfaces, and tunable model hyper-parameters.

Every magic number the model depends on lives here so that it can be tuned in
one place and logged alongside a trained artifact.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TENNISDASH_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TOURS = ("atp", "wta")

# Canonical surface vocabulary. "Carpet" is historical but still present in the
# archives, so it is kept rather than silently folded into Hard.
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

# Tournament levels, ordered by how much signal a result carries.
LEVEL_WEIGHT = {
    "G": 1.00,   # Grand Slam
    "M": 0.95,   # Masters 1000 / WTA 1000
    "A": 0.90,   # ATP/WTA Tour regular
    "F": 1.00,   # Tour Finals
    "O": 0.85,   # Olympics
    "D": 0.80,   # Davis / BJK Cup
    "C": 0.70,   # Challenger
    "S": 0.60,   # Satellite / ITF
    "Q": 0.70,   # Qualifying
}


@dataclass
class EloConfig:
    """FiveThirtyEight-style Elo with a match-count-dependent K factor.

    K_i = k_scale / (matches_played_i + k_shift) ** k_decay
    New players move fast; established players move slowly. This is materially
    better calibrated than a fixed K because it lets a rising junior's rating
    catch up within a season without making a veteran's rating jittery.
    """

    initial: float = 1500.0
    k_scale: float = 250.0
    k_shift: float = 5.0
    k_decay: float = 0.4
    # Multiplier applied to K so surface ratings (which see ~1/3 the matches)
    # still move at a sensible pace.
    surface_k_multiplier: float = 1.15
    # Weight on the surface-specific rating when blending with overall Elo.
    # 0 => pure overall, 1 => pure surface. Tuned per surface: clay and grass
    # are the most surface-idiosyncratic.
    surface_blend: dict = field(
        default_factory=lambda: {
            "Hard": 0.55,
            "Clay": 0.68,
            "Grass": 0.70,
            "Carpet": 0.60,
        }
    )
    # Margin-of-victory: scale the rating update by how dominant the scoreline
    # was, relative to what was expected. Bounded to avoid blowouts dominating.
    use_margin_of_victory: bool = True
    mov_min: float = 0.75
    mov_max: float = 1.35
    # Ratings regress toward the mean during long layoffs (injury, maternity,
    # suspension). Half-life in days of the "certainty" of a stale rating.
    layoff_grace_days: int = 90
    layoff_regression_per_year: float = 0.28
    # Cap total regression so a returning former #1 is not reset to average.
    layoff_regression_cap: float = 0.45
    # Weight of a match by tournament level (see LEVEL_WEIGHT).
    use_level_weight: bool = True
    # Retirements/walkovers carry partial information: the winner did win, but
    # the scoreline is uninformative.
    retirement_weight: float = 0.55
    walkover_weight: float = 0.0


@dataclass
class ServeReturnConfig:
    """Opponent-adjusted serve and return skill estimation.

    Raw serve-points-won is heavily schedule-contaminated: a player who spent
    the year losing to big returners looks worse than they are. We fit an
    additive model  spw_ij = mu_surface + serve_i - return_j  by ridge-penalised
    least squares over a rolling window, which removes that contamination.
    """

    # Rolling window of history used for each refit. Tuned by sweeping
    # out-of-sample serve-percentage RMSE (see `tennisdash tune`).
    window_days: int = 1095
    # Exponential half-life (days) applied to observations inside the window.
    half_life_days: float = 365.0
    # Ridge penalty, in units of *weighted service points*: a player needs
    # roughly `ridge_lambda` service points before their rating is pulled
    # halfway out of the tour average. Because one match supplies ~70
    # service points, a penalty of order 10 would shrink almost nothing -
    # this must be scaled in points, not matches. Higher => more shrinkage toward tour average, which is the
    # right call for players with few matches.
    ridge_lambda: float = 200.0
    # Service points below which a player's rating is flagged low-confidence
    # in the UI. Shrinkage itself is handled entirely by `ridge_lambda`.
    min_points: int = 250
    # Surfaces are fitted jointly with a surface offset plus a per-player
    # surface deviation shrunk by this extra penalty.
    surface_ridge_lambda: float = 550.0
    # Number of coordinate-descent sweeps for the ridge solve.
    max_iter: int = 60
    tol: float = 1e-7


@dataclass
class FeatureConfig:
    """Form, fatigue and context feature windows."""

    form_half_life_days: float = 120.0
    form_windows: tuple = (10, 25, 50)
    fatigue_windows_days: tuple = (7, 14, 28)
    # Rest curve: matches played after this many idle days start to show rust.
    rest_optimal_days: float = 5.0
    # H2H is mostly noise. Shrink toward the prior with this pseudo-count.
    h2h_prior_matches: float = 6.0
    # Clutch metrics (break points, tiebreaks, deciding sets) are noisy;
    # regress them toward tour average with these pseudo-counts.
    bp_prior: float = 60.0
    tiebreak_prior: float = 25.0
    decider_prior: float = 12.0
    # Elevation (metres) above which serve dominance measurably increases.
    altitude_reference_m: float = 500.0
    # Minimum matches before a player is included in the training set at all.
    min_career_matches: int = 10


@dataclass
class ModelConfig:
    """Ensemble + calibration settings."""

    gbm_max_iter: int = 400
    gbm_learning_rate: float = 0.045
    gbm_max_leaf_nodes: int = 24
    gbm_min_samples_leaf: int = 60
    gbm_l2_regularization: float = 1.2
    gbm_max_depth: int | None = 6
    gbm_early_stopping_rounds: int = 40
    # Fraction of the (time-ordered) training window held out for calibration
    # and for stacker fitting.
    holdout_fraction: float = 0.18
    # Calibration method: "isotonic" or "platt".
    calibration: str = "isotonic"
    # Isotonic needs a decent sample; fall back to Platt below this many rows.
    isotonic_min_samples: int = 4000
    random_state: int = 7


@dataclass
class Config:
    elo: EloConfig = field(default_factory=EloConfig)
    serve_return: ServeReturnConfig = field(default_factory=ServeReturnConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()
