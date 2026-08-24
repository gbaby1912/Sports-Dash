"""Leakage tests.

The single most dangerous failure mode for a sports model is a feature that
quietly encodes the result it is supposed to predict. It does not announce
itself - it just produces suspiciously good accuracy that evaporates in
production. These tests check the property directly rather than trusting that
the chronological passes were written correctly.
"""
import numpy as np
import pandas as pd
import pytest

from tennisdash.features.builder import (
    META_COLUMNS,
    antisymmetric_columns,
    feature_columns,
    flip_features,
)


def test_features_are_independent_of_the_match_outcome(small_matches):
    """Re-run the pipeline with every result flipped; features must not move.

    If any feature peeked at the current match, swapping winner and loser would
    change it. Ratings and history legitimately change *after* the flipped
    match, so only the very first match of each player pair is compared - the
    point at which no prior state differs between the two runs.
    """
    from tennisdash.features.elo import run_elo

    original = small_matches.copy()
    elo_original, _ = run_elo(original)

    # Build a mirrored table where winner and loser swap for one target match.
    target = 500
    flipped = original.copy()
    row = flipped.iloc[target]
    swap_columns = [c for c in flipped.columns if c.startswith("winner_")]
    for column in swap_columns:
        loser_column = column.replace("winner_", "loser_")
        flipped.iloc[target, flipped.columns.get_loc(column)] = row[loser_column]
        flipped.iloc[target, flipped.columns.get_loc(loser_column)] = row[column]

    elo_flipped, _ = run_elo(flipped)

    # Every pre-match rating for the target match must be identical, up to the
    # winner/loser relabelling.
    for name in ("elo", "elo_surface", "elo_points", "elo_games"):
        assert elo_original.iloc[target][f"w_{name}"] == pytest.approx(
            elo_flipped.iloc[target][f"l_{name}"]
        ), f"{name} changed when the result was flipped - it saw the outcome"
        assert elo_original.iloc[target][f"l_{name}"] == pytest.approx(
            elo_flipped.iloc[target][f"w_{name}"]
        )


def test_ratings_only_use_earlier_matches(small_matches):
    """A player's first-ever match must carry the default rating."""
    from tennisdash.features.elo import run_elo

    features, _ = run_elo(small_matches)
    seen: set[tuple] = set()
    checked = 0
    for position, row in enumerate(small_matches.itertuples(index=False)):
        for tag, player in (("w", row.winner_id), ("l", row.loser_id)):
            key = (row.tour, int(player))
            if key not in seen:
                assert features.iloc[position][f"{tag}_elo"] == pytest.approx(1500.0), (
                    "a player's debut match already carried a non-default rating"
                )
                assert features.iloc[position][f"{tag}_elo_matches"] == 0
                seen.add(key)
                checked += 1
    assert checked > 50


def test_history_counters_never_include_the_current_match(small_matches):
    """Career match counts must lag the row they describe by exactly one."""
    from tennisdash.features.elo import run_elo
    from tennisdash.features.history import run_history

    elo, _ = run_elo(small_matches)
    history, _ = run_history(small_matches, elo)

    counts: dict = {}
    for position, row in enumerate(small_matches.itertuples(index=False)):
        for tag, player in (("w", row.winner_id), ("l", row.loser_id)):
            key = (row.tour, int(player))
            expected = counts.get(key, 0)
            assert history.iloc[position][f"{tag}_career_matches"] == expected
            if not row.walkover:
                counts[key] = expected + 1


def test_serve_return_fit_excludes_the_evaluation_window(small_matches):
    """A fit made `as_of` a date must not move when later matches change."""
    from tennisdash.features.serve_return import fit_serve_return

    cut = pd.Timestamp("2020-01-01")
    baseline = fit_serve_return(small_matches, as_of=cut)

    corrupted = small_matches.copy()
    future = corrupted["match_date"] >= cut
    # Wreck every future serve stat. A leak-free fit cannot notice.
    corrupted.loc[future, "winner_spw"] = 1.0
    corrupted.loc[future, "loser_spw"] = 1.0
    corrupted.loc[future, "winner_svpt"] = 1.0
    corrupted.loc[future, "loser_svpt"] = 1.0

    after = fit_serve_return(corrupted, as_of=cut)
    assert set(baseline.serve) == set(after.serve)
    for key, value in baseline.serve.items():
        assert value == pytest.approx(after.serve[key], abs=1e-12)


def test_every_antisymmetric_feature_flips_exactly(small_features):
    """The flip contract must hold for every column, not just the d_ prefixed."""
    features, _ = small_features
    flipped = flip_features(features)
    antisymmetric = set(antisymmetric_columns(features.columns))

    for column in feature_columns(features):
        original = features[column].to_numpy(dtype=float)
        mirrored = flipped[column].to_numpy(dtype=float)
        finite = np.isfinite(original) & np.isfinite(mirrored)
        if column in antisymmetric:
            assert np.allclose(mirrored[finite], -original[finite], atol=1e-9), (
                f"{column} is declared antisymmetric but did not negate"
            )
        elif column == "markov_prob":
            assert np.allclose(mirrored[finite], 1 - original[finite], atol=1e-9)
        else:
            assert np.allclose(mirrored[finite], original[finite], atol=1e-9), (
                f"{column} changed under the player swap but is not declared antisymmetric"
            )


def test_labels_are_balanced_by_construction(small_features):
    """Orientation is randomised, so the label must be close to a coin flip."""
    features, _ = small_features
    rate = features["label"].mean()
    assert 0.45 < rate < 0.55


def test_meta_columns_never_reach_the_model(small_features):
    features, _ = small_features
    model_columns = set(feature_columns(features))
    assert not model_columns & set(META_COLUMNS)
    assert "label" not in model_columns
    for column in model_columns:
        assert features[column].dtype.kind in "fiub", f"{column} is not numeric"
