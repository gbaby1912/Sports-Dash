"""Elo and the opponent-adjusted serve/return model."""
import numpy as np
import pandas as pd
import pytest

from tennisdash.config import CONFIG
from tennisdash.features.elo import EloEngine, _stretch


class TestElo:
    def test_expected_is_symmetric(self):
        engine = EloEngine()
        assert engine.expected(1600, 1400) + engine.expected(1400, 1600) == pytest.approx(1.0)
        assert engine.expected(1500, 1500) == pytest.approx(0.5)

    def test_four_hundred_points_is_ten_to_one(self):
        """The defining property of the Elo scale."""
        engine = EloEngine()
        odds = engine.expected(1900, 1500) / (1 - engine.expected(1900, 1500))
        assert odds == pytest.approx(10.0, rel=1e-6)

    def test_k_factor_decays_with_experience(self):
        engine = EloEngine()
        assert engine._k_factor(0) > engine._k_factor(50) > engine._k_factor(500)

    def test_stretch_is_exactly_symmetric(self):
        for share in (0.5, 0.55, 0.62, 0.4):
            assert _stretch(share) + _stretch(1 - share) == pytest.approx(1.0, abs=1e-12)

    def test_layoff_regresses_toward_the_mean(self):
        engine = EloEngine()
        from tennisdash.features.elo import PlayerState

        state = PlayerState(rating=2000.0, matches=200,
                            last_date=pd.Timestamp("2020-01-01"))
        engine._apply_layoff(state, pd.Timestamp("2022-01-01"))
        assert 1500 < state.rating < 2000, "a two-year absence should cost rating"

        fresh = PlayerState(rating=2000.0, matches=200, last_date=pd.Timestamp("2020-01-01"))
        engine._apply_layoff(fresh, pd.Timestamp("2020-02-01"))
        assert fresh.rating == pytest.approx(2000.0), "a month off should cost nothing"

    def test_layoff_regression_is_capped(self):
        """A returning great is not reset to average, however long the absence."""
        engine = EloEngine()
        from tennisdash.features.elo import PlayerState

        state = PlayerState(rating=2200.0, matches=400, last_date=pd.Timestamp("2010-01-01"))
        engine._apply_layoff(state, pd.Timestamp("2024-01-01"))
        floor = 2200 - (2200 - 1500) * CONFIG.elo.layoff_regression_cap
        assert state.rating >= floor - 1e-6

    def test_winner_gains_and_loser_loses(self, small_matches):
        engine = EloEngine()
        row = small_matches.iloc[0].to_dict()
        before_w = engine.rating_of(row["tour"], row["winner_id"])
        engine.update(row)
        assert engine.rating_of(row["tour"], row["winner_id"]) > before_w
        assert engine.rating_of(row["tour"], row["loser_id"]) < before_w

    def test_blended_rating_falls_back_to_overall(self):
        """With no matches on a surface, the blend is exactly the overall rating."""
        from tennisdash.features.elo import PlayerState

        engine = EloEngine()
        engine._overall[("atp", 1)] = PlayerState(rating=1800.0, matches=120)
        assert engine.blended("atp", 1, "Grass") == pytest.approx(1800.0)

    def test_ratings_predict_results_better_than_chance(self, small_matches):
        from tennisdash.features.elo import run_elo

        features, _ = run_elo(small_matches)
        established = (features.w_elo_matches > 20) & (features.l_elo_matches > 20)
        gap = (features.w_elo_blend - features.l_elo_blend)[established]
        assert (gap > 0).mean() > 0.62


class TestServeReturn:
    def test_expected_spw_stays_in_a_plausible_band(self, small_matches):
        from tennisdash.features.serve_return import fit_serve_return

        fit = fit_serve_return(small_matches, as_of=pd.Timestamp("2021-06-01"))
        assert fit.serve, "no players were rated"
        players = [k[1] for k in list(fit.serve)[:20]]
        for server in players:
            for returner in players:
                value = fit.expected_spw("atp", server, returner, "Hard")
                assert 0.30 <= value <= 0.90

    def test_baselines_match_the_tour(self, small_matches):
        """Fitted baselines should land near the real serve averages."""
        from tennisdash.features.serve_return import fit_serve_return

        fit = fit_serve_return(small_matches, as_of=pd.Timestamp("2021-06-01"))
        atp_hard = 1 / (1 + np.exp(-fit.tour_baseline("atp", "Hard")))
        wta_hard = 1 / (1 + np.exp(-fit.tour_baseline("wta", "Hard")))
        assert 0.58 < atp_hard < 0.72
        assert 0.52 < wta_hard < 0.66
        assert atp_hard > wta_hard, "men hold serve more often than women"

    def test_shrinkage_pulls_thin_samples_toward_average(self, small_matches):
        """A player with few points must not be given an extreme rating."""
        from tennisdash.features.serve_return import fit_serve_return

        fit = fit_serve_return(small_matches, as_of=pd.Timestamp("2021-06-01"))
        thin = [k for k, v in fit.serve_points.items() if 0 < v < 200]
        thick = [k for k, v in fit.serve_points.items() if v > 3000]
        if thin and thick:
            thin_spread = np.std([abs(fit.serve[k]) for k in thin])
            thick_spread = np.std([abs(fit.serve[k]) for k in thick])
            assert thin_spread < thick_spread * 1.6

    def test_rolling_fits_are_reproducible(self, small_matches):
        from tennisdash.features.serve_return import RollingServeReturn

        first = RollingServeReturn(refit_days=90).build(small_matches)
        second = RollingServeReturn(refit_days=90).build(small_matches)
        pd.testing.assert_frame_equal(first, second)
