"""Live prediction: score a hypothetical matchup.

The whole point of this module is that it does **not** contain a second feature
builder. It synthesises a match row in the archive's own winner/loser layout,
runs it through the identical Elo, serve/return and history accessors used
during training, and calls the same ``assemble``. Train/serve skew - where the
served model quietly sees slightly different features from the trained one - is
one of the most common ways a good model becomes a bad product, and the only
reliable defence is to have exactly one implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LEVEL_WEIGHT
from .data.venues import is_indoor, venue_altitude
from .features.builder import assemble, feature_columns
from .features.serve_return import _row_features as serve_return_row
from .models.markov import match_win_probability, score_distribution
from .players import age_on


@dataclass
class MatchContext:
    """Everything about the match that is not about the two players."""

    surface: str = "Hard"
    best_of: int = 3
    level: str = "A"
    round: str = "R32"
    tourney_name: str = "Neutral Court"
    indoor: bool | None = None
    altitude_m: int | None = None
    draw_size: int = 32
    # None means "the end of the model's data" - see MatchPredictor.as_of.
    date: pd.Timestamp | None = None

    def resolved_indoor(self) -> bool:
        if self.indoor is not None:
            return bool(self.indoor)
        return is_indoor(self.tourney_name, self.surface)

    def resolved_altitude(self) -> int:
        if self.altitude_m is not None:
            return int(self.altitude_m)
        return venue_altitude(self.tourney_name)


@dataclass
class Prediction:
    p1_id: int
    p2_id: int
    p1_name: str
    p2_name: str
    probability: float
    base_models: dict = field(default_factory=dict)
    serve: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    ratings: dict = field(default_factory=dict)
    factors: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "p1": {"id": self.p1_id, "name": self.p1_name},
            "p2": {"id": self.p2_id, "name": self.p2_name},
            "p1_win_probability": self.probability,
            "p2_win_probability": 1.0 - self.probability,
            "base_models": self.base_models,
            "serve": self.serve,
            "score_distribution": self.scores,
            "ratings": self.ratings,
            "factors": self.factors,
            "factor_groups": self.groups,
        }


class MatchPredictor:
    """Wraps a trained bundle and answers matchup questions."""

    def __init__(self, bundle: dict) -> None:
        self.ensemble = bundle["ensemble"]
        self.elo = bundle["engines"]["elo"]
        self.rolling = bundle["engines"]["serve_return"]
        self.history = bundle["engines"]["history"]
        self.directory = bundle["directory"]
        self.metadata = bundle.get("metadata", {})
        # Default "now" for a prediction is the end of the training data, not the
        # wall clock. With a live, current dataset the two coincide. With an
        # archive that ends months ago they do not, and using today would hand
        # every player a huge fake layoff - a value far outside anything in
        # training, which the fatigue features then extrapolate wildly from. The
        # symptom is a prediction dominated by "days since last match" for two
        # players who are both simply absent from the end of the data.
        span = (self.metadata.get("data_span") or [None, None])[1]
        self.as_of = pd.Timestamp(span) if span else pd.Timestamp.today().normalize()
        self._index = {
            (row.tour, int(row.player_id)): row
            for row in self.directory.itertuples(index=False)
        }

    # ------------------------------------------------------------- lookups
    def player(self, tour: str, player_id: int):
        key = (tour, int(player_id))
        if key not in self._index:
            raise KeyError(f"unknown player {player_id} on tour {tour}")
        return self._index[key]

    def find(self, tour: str, query: str, limit: int = 12) -> pd.DataFrame:
        """Case-insensitive substring search over the player directory."""
        block = self.directory[self.directory["tour"] == tour]
        matched = block[block["name"].str.contains(query, case=False, na=False, regex=False)]
        return matched.sort_values("matches", ascending=False).head(limit)

    # ------------------------------------------------------------ features
    def _synthetic_row(
        self, tour: str, p1: int, p2: int, context: MatchContext
    ) -> dict:
        """Build one archive-layout match row with p1 in the winner slot."""
        date = pd.Timestamp(context.date) if context.date is not None else self.as_of
        row: dict = {
            "match_id": f"live-{tour}-{p1}-{p2}",
            "tour": tour,
            "tourney_id": "live",
            "tourney_name": context.tourney_name,
            "tourney_date": date,
            "match_date": date,
            "match_num": 0,
            "surface": context.surface,
            "indoor": context.resolved_indoor(),
            "altitude_m": context.resolved_altitude(),
            "draw_size": context.draw_size,
            "level": context.level,
            "level_weight": LEVEL_WEIGHT.get(context.level, 0.9),
            "round": context.round,
            "best_of": int(context.best_of),
            "minutes": np.nan,
            "score": "",
            "retirement": False,
            "walkover": False,
            "completed": True,
            "sets_played": 0,
            "went_to_decider": False,
            "tiebreaks_played": 0,
            "dominance": 0.5,
            "has_serve_stats": True,
        }
        for slot, player_id in (("winner", p1), ("loser", p2)):
            info = self.player(tour, player_id)
            row[f"{slot}_id"] = int(player_id)
            row[f"{slot}_name"] = info.name
            row[f"{slot}_hand"] = info.hand
            row[f"{slot}_height_cm"] = info.height_cm
            row[f"{slot}_ioc"] = info.ioc
            row[f"{slot}_age"] = age_on(pd.Series(info._asdict()), date)
            row[f"{slot}_rank"] = info.last_rank
            row[f"{slot}_rank_points"] = info.last_rank_points
        return row

    def build_feature_row(
        self, tour: str, p1: int, p2: int, context: MatchContext
    ) -> pd.DataFrame:
        """Feature matrix for a single hypothetical match, p1 as player 1."""
        row = self._synthetic_row(tour, p1, p2, context)
        combined = dict(row)
        combined.update(self.elo.pre_match_features(row))
        fit = self.rolling.latest
        if fit is not None:
            combined.update(serve_return_row(fit, row))
        combined.update(self.history.pre_match_features(row))
        frame = pd.DataFrame([combined])
        features = assemble(frame, p1_is_winner=np.array([True]))
        # Keep the row inside the range the model actually saw. See
        # TennisEnsemble._compute_bounds for why a hypothetical matchup needs
        # this and a row from the training set does not.
        return self.ensemble.clamp(features)

    # ------------------------------------------------------------- predict
    def predict(
        self, tour: str, p1: int, p2: int, context: MatchContext | None = None
    ) -> Prediction:
        context = context or MatchContext()
        features = self.build_feature_row(tour, p1, p2, context)
        probability = float(self.ensemble.predict(features)[0])

        p1_info, p2_info = self.player(tour, p1), self.player(tour, p2)
        fit = self.rolling.latest
        surface = context.surface

        serve: dict = {}
        scores: dict = {}
        if fit is not None:
            p1_spw = fit.expected_spw(tour, p1, p2, surface)
            p2_spw = fit.expected_spw(tour, p2, p1, surface)
            from .models.markov import game_win_probability

            serve = {
                "p1_expected_spw": p1_spw,
                "p2_expected_spw": p2_spw,
                "p1_hold_pct": float(game_win_probability(p1_spw)[0]),
                "p2_hold_pct": float(game_win_probability(p2_spw)[0]),
                "p1_serve_skill": fit.serve_skill(tour, p1, surface),
                "p2_serve_skill": fit.serve_skill(tour, p2, surface),
                "p1_return_skill": fit.return_skill(tour, p1, surface),
                "p2_return_skill": fit.return_skill(tour, p2, surface),
                "p1_raw_spw": fit.raw_spw.get((tour, p1)),
                "p2_raw_spw": fit.raw_spw.get((tour, p2)),
                "p1_raw_rpw": fit.raw_rpw.get((tour, p1)),
                "p2_raw_rpw": fit.raw_rpw.get((tour, p2)),
                "markov_probability": float(
                    match_win_probability(p1_spw, p2_spw, best_of=context.best_of)[0]
                ),
            }
            scores = _rescale_scores(
                score_distribution(p1_spw, p2_spw, best_of=context.best_of), probability
            )

        ratings = {}
        for tag, player_id in (("p1", p1), ("p2", p2)):
            ratings[tag] = {
                "elo": self.elo.rating_of(tour, player_id, "overall"),
                "elo_surface": self.elo.blended(tour, player_id, surface),
                "elo_points": self.elo.rating_of(tour, player_id, "points"),
                "peak_elo": self.elo.peak(tour, player_id),
                "matches": self.elo.matches_played(tour, player_id),
            }

        return Prediction(
            p1_id=int(p1),
            p2_id=int(p2),
            p1_name=p1_info.name,
            p2_name=p2_info.name,
            probability=probability,
            base_models={
                name: float(value[0])
                for name, value in self.ensemble.base_predictions(features).items()
            },
            serve=serve,
            scores=scores,
            ratings=ratings,
            factors=self.explain(features, probability),
            groups=self.explain_groups(features, probability),
            features={
                c: (None if pd.isna(v) else float(v))
                for c, v in features[feature_columns(features)].iloc[0].items()
            },
        )

    # ------------------------------------------------------------- explain
    def _occlusion(
        self,
        features: pd.DataFrame,
        probability: float,
        variants: list[tuple[str, list[str]]],
    ) -> dict[str, float]:
        """Score many "what if these two were level here?" variants at once.

        Each variant zeroes one set of antisymmetric features - making the two
        players equal on that dimension - and the shift in log-odds is that
        set's contribution. Zero is the neutral value by construction, because
        every antisymmetric feature is a difference between the two sides.

        The variants are stacked into a single frame and scored in one call.
        Done one at a time this is ~60 separate model evaluations per
        prediction, and the fixed per-call overhead of four scikit-learn
        pipelines dominates completely - batching turns a multi-second response
        into a few milliseconds.
        """
        if not variants:
            return {}
        baseline_logit = _logit(probability)
        batch = pd.concat([features] * len(variants), ignore_index=True)
        for position, (_, columns) in enumerate(variants):
            for column in columns:
                batch.iloc[position, batch.columns.get_loc(column)] = 0.0

        probabilities = self.ensemble.predict(batch)
        return {
            name: baseline_logit - _logit(float(value))
            for (name, _), value in zip(variants, probabilities)
        }

    def explain_groups(self, features: pd.DataFrame, probability: float) -> list[dict]:
        """Attribute the prediction to *categories* of evidence.

        Individual features in this model are heavily correlated - overall Elo,
        surface Elo, peak Elo and games-Elo all move together - so neutralising
        them one at a time splits one real effect into six identical-looking
        small ones and understates all of them. Neutralising a whole category at
        once measures the thing a user actually wants to know: how much of the
        edge comes from ratings, versus serve/return, versus form, versus rest.
        """
        row = features.iloc[0]
        variants = []
        for group, columns in FEATURE_GROUPS.items():
            present = [
                c for c in columns
                if c in features.columns and pd.notna(row.get(c)) and float(row[c]) != 0.0
            ]
            if present:
                variants.append((group, present))

        contributions = self._occlusion(features, probability, variants)
        rows = []
        for group, columns in variants:
            contribution = contributions.get(group, 0.0)
            if abs(contribution) < 1e-9:
                continue
            rows.append(
                {
                    "group": group,
                    "label": GROUP_LABELS.get(group, group),
                    "description": GROUP_NOTES.get(group, ""),
                    "contribution": float(contribution),
                    "favours": "p1" if contribution > 0 else "p2",
                    "n_features": len(columns),
                }
            )
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows

    def explain(self, features: pd.DataFrame, probability: float, top: int = 8) -> list[dict]:
        """Attribute the prediction to individual features.

        Same occlusion idea as :meth:`explain_groups`, one feature at a time.
        Useful detail, but read it alongside the group view: because several of
        these features move together, each looks smaller on its own than the
        category it belongs to.
        """
        from .features.builder import antisymmetric_columns

        row = features.iloc[0]
        variants = [
            (column, [column])
            for column in antisymmetric_columns(self.ensemble.columns)
            if pd.notna(row.get(column)) and float(row[column]) != 0.0
        ]
        contributions = self._occlusion(features, probability, variants)

        rows = []
        for column, _ in variants:
            contribution = contributions.get(column, 0.0)
            if abs(contribution) < 1e-6:
                continue
            rows.append(
                {
                    "feature": column,
                    "label": FEATURE_LABELS.get(column, column),
                    "value": float(row[column]),
                    "contribution": float(contribution),
                    "favours": "p1" if contribution > 0 else "p2",
                }
            )
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows[:top]


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _rescale_scores(distribution: dict[str, float], probability: float) -> dict[str, float]:
    """Reconcile the Markov score distribution with the ensemble's win probability.

    The Markov model supplies the *shape* of the scoreline distribution, which it
    is good at, while the ensemble supplies the win probability, which it is
    better at. Rescaling p1's and p2's branches separately makes the two agree
    instead of showing a user a 62% win probability next to scorelines summing
    to 55%.
    """
    p1_keys = [k for k in distribution if int(k.split("-")[0]) > int(k.split("-")[1])]
    p2_keys = [k for k in distribution if k not in p1_keys]
    p1_mass = sum(distribution[k] for k in p1_keys)
    p2_mass = sum(distribution[k] for k in p2_keys)
    out = {}
    for key in p1_keys:
        out[key] = distribution[key] / p1_mass * probability if p1_mass > 0 else 0.0
    for key in p2_keys:
        out[key] = distribution[key] / p2_mass * (1 - probability) if p2_mass > 0 else 0.0
    return out


FEATURE_LABELS = {
    "d_elo": "Overall Elo gap",
    "d_elo_surface": "Surface Elo gap",
    "d_elo_blend": "Surface-weighted Elo gap",
    "d_elo_points": "Points-won Elo gap",
    "d_elo_games": "Games-won Elo gap",
    "d_elo_peak": "Career-peak Elo gap",
    "d_elo_vs_peak": "Distance below own peak",
    "d_serve_skill": "Serve skill (opponent-adjusted)",
    "d_return_skill": "Return skill (opponent-adjusted)",
    "d_serve_skill_surface": "Serve skill on this surface",
    "d_return_skill_surface": "Return skill on this surface",
    "d_serve_dominance": "Serve/return matchup edge",
    "d_exp_spw": "Expected serve points won gap",
    "d_form_decayed": "Recent form (time-decayed)",
    "d_form_quality": "Quality of recent opposition",
    "d_form_win_pct_10": "Win rate, last 10",
    "d_form_win_pct_25": "Win rate, last 25",
    "d_form_win_pct_50": "Win rate, last 50",
    "d_minutes_7d": "Court time, last 7 days",
    "d_minutes_14d": "Court time, last 14 days",
    "d_minutes_28d": "Court time, last 28 days",
    "d_matches_7d": "Matches, last 7 days",
    "d_matches_14d": "Matches, last 14 days",
    "d_sets_14d": "Sets played, last 14 days",
    "d_days_since_last": "Days since last match",
    "d_rest_deviation": "Rest vs optimal",
    "d_is_returning": "Returning from a layoff",
    "d_recent_retirements": "Recent retirements (injury proxy)",
    "d_bp_saved_pct": "Break points saved",
    "d_bp_conv_pct": "Break points converted",
    "d_tiebreak_pct": "Tiebreak record",
    "d_decider_pct": "Deciding-set record",
    "d_surface_win_pct": "Career win rate on this surface",
    "d_surface_matches": "Experience on this surface",
    "d_surface_recency": "Recent time on this surface",
    "d_career_win_pct": "Career win rate",
    "d_career_matches": "Career match count",
    "d_win_streak": "Current win streak",
    "d_loss_streak": "Current losing streak",
    "d_h2h_win_pct": "Head-to-head record",
    "d_h2h_surface_win_pct": "Head-to-head on this surface",
    "d_age": "Age gap",
    "d_age_from_peak": "Distance from peak age",
    "d_height_cm": "Height gap",
    "d_log_rank": "Ranking gap",
    "d_log_rank_points": "Ranking points gap",
    "lefty_edge": "Left-hander matchup",
    "home_edge": "Home crowd",
    "markov_logit": "Point-model projection",
}


# Categories of evidence, used for group-level attribution. Every antisymmetric
# feature belongs to exactly one group, so the categories partition the evidence
# rather than overlapping.
FEATURE_GROUPS: dict[str, list[str]] = {
    "ratings": [
        "d_elo", "d_elo_surface", "d_elo_blend", "d_elo_points", "d_elo_games",
        "d_elo_peak", "d_elo_vs_peak",
    ],
    "serve_return": [
        "d_serve_skill", "d_return_skill", "d_serve_skill_surface",
        "d_return_skill_surface", "d_serve_dominance", "d_exp_spw", "markov_logit",
    ],
    "surface": [
        "d_surface_win_pct", "d_surface_matches", "d_surface_recency",
    ],
    "form": [
        "d_form_decayed", "d_form_quality", "d_form_win_pct_10",
        "d_form_win_pct_25", "d_form_win_pct_50", "d_win_streak", "d_loss_streak",
    ],
    "fatigue": [
        "d_minutes_7d", "d_minutes_14d", "d_minutes_28d",
        "d_matches_7d", "d_matches_14d", "d_matches_28d",
        "d_sets_7d", "d_sets_14d", "d_sets_28d",
        "d_days_since_last", "d_rest_deviation", "d_is_returning",
        "d_recent_retirements",
    ],
    "clutch": [
        "d_bp_saved_pct", "d_bp_conv_pct", "d_tiebreak_pct", "d_decider_pct",
    ],
    "h2h": ["d_h2h_win_pct", "d_h2h_surface_win_pct"],
    "experience": ["d_career_matches", "d_career_win_pct", "d_log_rank", "d_log_rank_points"],
    "physical": ["d_age", "d_age_from_peak", "d_height_cm", "lefty_edge"],
    "context": ["home_edge"],
}

GROUP_LABELS = {
    "ratings": "Ratings",
    "serve_return": "Serve & return",
    "surface": "Surface record",
    "form": "Recent form",
    "fatigue": "Rest & workload",
    "clutch": "Clutch record",
    "h2h": "Head-to-head",
    "experience": "Ranking & experience",
    "physical": "Physical & style",
    "context": "Home advantage",
}

GROUP_NOTES = {
    "ratings": "Elo family: overall, surface, points-won, games-won and career peak.",
    "serve_return": "Opponent-adjusted serve/return skill and the point-model projection.",
    "surface": "Career record and recent time spent on this surface.",
    "form": "Time-decayed recent results, weighted by the quality of the opposition.",
    "fatigue": "Court time and matches in the last 7/14/28 days, plus rest since the last match.",
    "clutch": "Break points, tiebreaks and deciding sets - all shrunk toward tour average.",
    "h2h": "Prior meetings, shrunk hard toward even because most pairs have played rarely.",
    "experience": "Official ranking, ranking points and career match count.",
    "physical": "Age, height and the left/right-hander matchup.",
    "context": "Playing in front of a home crowd.",
}
