"""Leaderboard semantics.

Two things a naive Elo leaderboard gets wrong, both of which mislead rather than
merely inconvenience the reader:

* Elo only changes when a match is played, so a retired player keeps the rating
  they walked away with and can outrank everyone currently competing.
* A serve/return skill of 0.0 means "exactly tour average", but an unrated player
  also reads as 0.0 unless the two cases are kept apart.
"""
import pandas as pd
import pytest

from tennisdash.features.elo import EloEngine, PlayerState
from tennisdash.features.serve_return import fit_serve_return


class TestActiveFilter:
    def _engine(self):
        engine = EloEngine()
        engine._overall[("atp", 1)] = PlayerState(
            rating=2300.0, matches=500, last_date=pd.Timestamp("2012-06-01")
        )
        engine._overall[("atp", 2)] = PlayerState(
            rating=2100.0, matches=400, last_date=pd.Timestamp("2024-09-01")
        )
        engine._overall[("atp", 3)] = PlayerState(
            rating=2000.0, matches=300, last_date=pd.Timestamp("2024-10-01")
        )
        return engine

    def test_unfiltered_snapshot_includes_everyone(self):
        snapshot = self._engine().snapshot("atp")
        assert set(snapshot.player_id) == {1, 2, 3}
        assert snapshot.iloc[0].player_id == 1, "highest rating should lead"

    def test_active_filter_drops_the_long_retired(self):
        snapshot = self._engine().snapshot("atp", active_since=pd.Timestamp("2024-01-01"))
        assert set(snapshot.player_id) == {2, 3}
        assert snapshot.iloc[0].player_id == 2

    def test_players_with_no_recorded_date_are_dropped_when_filtering(self):
        engine = self._engine()
        engine._overall[("atp", 4)] = PlayerState(rating=2500.0, matches=10, last_date=None)
        snapshot = engine.snapshot("atp", active_since=pd.Timestamp("2024-01-01"))
        assert 4 not in set(snapshot.player_id)

    def test_snapshot_is_sorted_by_the_requested_surface(self):
        engine = self._engine()
        engine._surface[("atp", 3, "Clay")] = PlayerState(rating=2600.0, matches=200)
        overall = engine.snapshot("atp")
        clay = engine.snapshot("atp", surface="Clay")
        assert overall.iloc[0].player_id == 1
        assert clay.iloc[0].player_id == 3, "the clay specialist should lead on clay"

    def test_other_tours_are_excluded(self):
        engine = self._engine()
        engine._overall[("wta", 9)] = PlayerState(rating=2400.0, matches=100,
                                                  last_date=pd.Timestamp("2024-10-01"))
        assert 9 not in set(engine.snapshot("atp").player_id)


class TestRatingCoverage:
    def test_has_rating_separates_unknown_from_average(self, small_matches):
        fit = fit_serve_return(small_matches, as_of=pd.Timestamp("2021-06-01"))
        known = next(iter(fit.serve))
        assert fit.has_rating(known[0], known[1])
        assert not fit.has_rating("atp", 987654321)
        # The arithmetic default is still 0.0 - that is what makes the flag needed.
        assert fit.serve_skill("atp", 987654321) == 0.0

    def test_coverage_reports_zero_for_an_unknown_player(self, small_matches):
        fit = fit_serve_return(small_matches, as_of=pd.Timestamp("2021-06-01"))
        assert fit.coverage("atp", 987654321) == 0.0
