"""Ensemble, calibration and backtest metrics."""
import numpy as np
import pytest

from tennisdash.backtest.metrics import (
    accuracy,
    brier,
    calibration_slope,
    evaluate,
    expected_calibration_error,
    log_loss_safe,
    skill_score,
    betting_return,
)
from tennisdash.models.ensemble import Calibrator, TennisEnsemble


class TestMetrics:
    def test_perfect_prediction_scores_zero_loss(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        assert log_loss_safe(y, p) == pytest.approx(0.0, abs=1e-10)
        assert brier(y, p) == pytest.approx(0.0)

    def test_coin_flip_scores_log_two(self):
        y = np.random.default_rng(0).integers(0, 2, 5000)
        assert log_loss_safe(y, np.full(5000, 0.5)) == pytest.approx(np.log(2), abs=1e-9)

    def test_calibration_slope_detects_overconfidence(self):
        rng = np.random.default_rng(1)
        n = 40000
        truth = rng.beta(2, 2, n)
        y = (rng.random(n) < truth).astype(int)
        assert calibration_slope(y, truth) == pytest.approx(1.0, abs=0.05)
        stretched = np.clip(0.5 + (truth - 0.5) * 1.8, 0.01, 0.99)
        assert calibration_slope(y, stretched) < 0.7
        squashed = 0.5 + (truth - 0.5) * 0.5
        assert calibration_slope(y, squashed) > 1.4

    def test_accuracy_is_blind_to_confidence(self):
        """The reason accuracy is not the headline metric."""
        rng = np.random.default_rng(2)
        n = 20000
        truth = rng.beta(2, 2, n)
        y = (rng.random(n) < truth).astype(int)
        overconfident = np.clip(0.5 + (truth - 0.5) * 1.9, 0.01, 0.99)
        assert accuracy(y, overconfident) == pytest.approx(accuracy(y, truth))
        assert log_loss_safe(y, overconfident) > log_loss_safe(y, truth)

    def test_ece_is_zero_for_a_calibrated_model(self):
        rng = np.random.default_rng(3)
        truth = rng.beta(2, 2, 60000)
        y = (rng.random(60000) < truth).astype(int)
        assert expected_calibration_error(y, truth) < 0.01

    def test_skill_score_sign(self):
        y = np.array([1] * 80 + [0] * 20)
        good = np.full(100, 0.8)
        bad = np.full(100, 0.5)
        assert skill_score(y, good, bad) > 0
        assert skill_score(y, bad, good) < 0

    def test_betting_return_pays_out_correctly(self):
        """A model that knows the truth against mispriced odds must profit."""
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        odds = np.full(4, 2.0)
        result = betting_return(y, p, odds, odds, edge_threshold=0.0, stake="flat")
        assert result["bets"] == 4
        assert result["profit"] == pytest.approx(4.0)

    def test_betting_return_handles_missing_odds(self):
        result = betting_return(np.array([1, 0]), np.array([.6, .4]),
                                np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
        assert result["bets"] == 0


class TestCalibrator:
    def test_isotonic_corrects_a_biased_model(self):
        rng = np.random.default_rng(4)
        n = 20000
        truth = rng.beta(2, 2, n)
        y = (rng.random(n) < truth).astype(int)
        biased = np.clip(truth * 0.6 + 0.2, 0.01, 0.99)
        calibrator = Calibrator("isotonic").fit(biased, y)
        corrected = calibrator.transform(biased)
        assert expected_calibration_error(y, corrected) < expected_calibration_error(y, biased)

    def test_platt_is_used_when_the_sample_is_small(self):
        rng = np.random.default_rng(5)
        p = rng.random(300)
        y = (rng.random(300) < p).astype(int)
        calibrator = Calibrator("isotonic").fit(p, y, min_isotonic=4000)
        assert calibrator.method == "platt"

    def test_unfitted_calibrator_is_the_identity(self):
        p = np.array([0.2, 0.5, 0.8])
        assert np.allclose(Calibrator().transform(p), p)


class TestEnsemble:
    @pytest.fixture(scope="class")
    def trained(self, small_features):
        features, _ = small_features
        split = features["match_date"] < features["match_date"].quantile(0.75)
        model = TennisEnsemble().fit(features[split], verbose=False)
        return model, features[~split]

    def test_predictions_are_exactly_antisymmetric(self, trained):
        model, test = trained
        from tennisdash.features.builder import flip_features

        forward = model.predict(test)
        backward = model.predict(flip_features(test))
        assert np.max(np.abs(forward + backward - 1.0)) < 1e-9

    def test_predictions_are_valid_probabilities(self, trained):
        model, test = trained
        p = model.predict(test)
        assert np.isfinite(p).all()
        assert (p > 0).all() and (p < 1).all()

    def test_beats_a_coin_flip_out_of_sample(self, trained):
        model, test = trained
        y = test["label"].to_numpy()
        assert log_loss_safe(y, model.predict(test)) < np.log(2) - 0.05

    def test_beats_plain_elo_out_of_sample(self, trained):
        """The ensemble has to earn its complexity against the real baseline."""
        model, test = trained
        y = test["label"].to_numpy()
        elo_only = 1 / (1 + 10 ** (-test["d_elo_blend"].fillna(0).to_numpy() / 400))
        assert log_loss_safe(y, model.predict(test)) < log_loss_safe(y, elo_only)

    def test_is_reasonably_calibrated(self, trained):
        model, test = trained
        y = test["label"].to_numpy()
        metrics = evaluate(y, model.predict(test))
        assert metrics["ece"] < 0.05
        assert 0.75 < metrics["calibration_slope"] < 1.3

    def test_every_base_learner_contributes_a_signal(self, trained):
        model, test = trained
        y = test["label"].to_numpy()
        for name, probability in model.base_predictions(test).items():
            assert log_loss_safe(y, probability) < np.log(2), f"{name} is worse than a coin"

    def test_metadata_is_recorded(self, trained):
        model, _ = trained
        assert model.training_rows > 0
        assert model.trained_through is not None
        assert set(model.stacker_weights) == {"rating", "markov", "gbm", "linear"}
        assert model.holdout_metrics["n"] > 0


class TestWalkForward:
    def test_folds_never_train_on_the_future(self, small_features):
        from tennisdash.backtest.walkforward import walk_forward

        features, _ = small_features
        summary, folds = walk_forward(features, min_train_matches=800, verbose=False)
        assert not summary.empty
        predictions = walk_forward.predictions
        assert predictions["match_id"].is_unique
        for row in summary[summary.tour == "all"].itertuples():
            assert 0.0 < row.log_loss < 1.0
            assert row.n > 0

    def test_reference_models_are_produced(self, small_features):
        from tennisdash.backtest.walkforward import reference_predictions

        features, _ = small_features
        references = reference_predictions(features)
        assert set(references) >= {"coin", "favourite", "elo"}
        for name, values in references.items():
            assert np.isfinite(values).all(), name
            assert ((values > 0) & (values < 1)).all(), name


class TestFeatureClamping:
    """Live prediction must not extrapolate outside the training envelope.

    A hypothetical matchup is not drawn from the same distribution as a real
    scheduled match: a user can pick two players who last competed a year apart,
    which essentially never happens inside a live draw. Unclamped, the linear
    members of the ensemble extrapolate from that happily and badly.
    """

    @pytest.fixture(scope="class")
    def trained(self, small_features):
        features, _ = small_features
        return TennisEnsemble().fit(features, verbose=False), features

    def test_bounds_are_recorded_for_every_usable_feature(self, trained):
        model, features = trained
        assert len(model.feature_bounds) > 50
        for column, (low, high) in model.feature_bounds.items():
            assert column in model.columns
            assert np.isfinite(low) and np.isfinite(high) and high > low

    def test_antisymmetric_bounds_are_symmetric(self, trained):
        """Otherwise clip(-x) stops equalling -clip(x) and symmetry breaks."""
        from tennisdash.features.builder import antisymmetric_columns

        model, _ = trained
        antisymmetric = set(antisymmetric_columns(model.columns))
        for column, (low, high) in model.feature_bounds.items():
            if column in antisymmetric:
                assert low + high == pytest.approx(0.0, abs=1e-12), column

    def test_clamp_pulls_extreme_values_into_range(self, trained):
        model, features = trained
        row = features.head(1).copy()
        column = "d_days_since_last"
        row[column] = 100000.0
        clamped = model.clamp(row)
        low, high = model.feature_bounds[column]
        assert clamped[column].iloc[0] == pytest.approx(high)
        assert low <= clamped[column].iloc[0] <= high

    def test_clamp_leaves_in_range_values_untouched(self, trained):
        model, features = trained
        sample = features.head(50)
        clamped = model.clamp(sample[model.columns])
        # 1st-99th percentile, so the vast majority of real rows are unchanged.
        unchanged = np.isclose(
            clamped.to_numpy(dtype=float), sample[model.columns].to_numpy(dtype=float),
            equal_nan=True,
        )
        assert unchanged.mean() > 0.93

    def test_clamping_preserves_exact_antisymmetry(self, trained):
        from tennisdash.features.builder import flip_features

        model, features = trained
        extreme = features.head(200).copy()
        for column in ("d_days_since_last", "d_elo", "d_minutes_28d"):
            if column in extreme.columns:
                extreme[column] = extreme[column] * 50.0
        clamped = model.clamp(extreme)
        forward = model.predict(clamped)
        backward = model.predict(flip_features(clamped))
        assert np.max(np.abs(forward + backward - 1.0)) < 1e-9

    def test_a_model_without_bounds_clamps_to_a_no_op(self, trained):
        model, features = trained
        model = TennisEnsemble(**{})
        assert model.clamp(features) is features
