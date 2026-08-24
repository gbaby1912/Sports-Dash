"""End-to-end: train a small bundle, serve it, exercise every endpoint.

This is the test that catches train/serve skew - the class of bug where the
model is fine but the thing actually answering requests is subtly different.
"""
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tennisdash.models.ensemble import TennisEnsemble  # noqa: E402
from tennisdash.players import build_directory  # noqa: E402
from tennisdash.predict import MatchContext, MatchPredictor  # noqa: E402


@pytest.fixture(scope="module")
def bundle(small_matches, small_features):
    features, engines = small_features
    ensemble = TennisEnsemble().fit(features, verbose=False)
    return {
        "ensemble": ensemble,
        "engines": engines,
        "directory": build_directory(small_matches),
        "metadata": {
            "training_rows": len(features),
            "n_features": len(ensemble.columns),
            "data_span": [str(small_matches.match_date.min()), str(small_matches.match_date.max())],
            "stacker_weights": ensemble.stacker_weights,
            "report": {},
        },
    }


@pytest.fixture(scope="module")
def predictor(bundle):
    return MatchPredictor(bundle)


@pytest.fixture(scope="module")
def two_players(predictor):
    block = predictor.directory[predictor.directory.tour == "atp"]
    block = block.sort_values("matches", ascending=False)
    return int(block.iloc[0].player_id), int(block.iloc[1].player_id)


class TestPredictor:
    def test_prediction_is_a_valid_probability(self, predictor, two_players):
        a, b = two_players
        result = predictor.predict("atp", a, b, MatchContext(surface="Hard"))
        assert 0.0 < result.probability < 1.0
        assert result.p1_id == a and result.p2_id == b

    def test_swapping_players_swaps_the_probability_exactly(self, predictor, two_players):
        a, b = two_players
        context = MatchContext(surface="Clay", best_of=5, level="G")
        forward = predictor.predict("atp", a, b, context).probability
        backward = predictor.predict("atp", b, a, context).probability
        assert forward + backward == pytest.approx(1.0, abs=1e-9)

    def test_score_distribution_agrees_with_the_win_probability(self, predictor, two_players):
        a, b = two_players
        result = predictor.predict("atp", a, b, MatchContext(best_of=3))
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)
        p1_mass = sum(v for k, v in result.scores.items()
                      if int(k.split("-")[0]) > int(k.split("-")[1]))
        assert p1_mass == pytest.approx(result.probability, abs=1e-6)

    def test_best_of_five_moves_the_probability_away_from_even(self, predictor, two_players):
        a, b = two_players
        bo3 = predictor.predict("atp", a, b, MatchContext(best_of=3)).probability
        bo5 = predictor.predict("atp", a, b, MatchContext(best_of=5)).probability
        assert abs(bo5 - 0.5) >= abs(bo3 - 0.5) - 0.02

    def test_surface_changes_the_answer(self, predictor, two_players):
        a, b = two_players
        values = {
            surface: predictor.predict("atp", a, b, MatchContext(surface=surface)).probability
            for surface in ("Hard", "Clay", "Grass")
        }
        assert len(set(np.round(list(values.values()), 6))) > 1, "surface had no effect"

    def test_factor_groups_partition_and_are_signed(self, predictor, two_players):
        a, b = two_players
        result = predictor.predict("atp", a, b, MatchContext())
        assert result.groups
        for group in result.groups:
            assert group["favours"] in ("p1", "p2")
            assert (group["contribution"] > 0) == (group["favours"] == "p1")

    def test_unknown_player_raises(self, predictor):
        with pytest.raises(KeyError):
            predictor.predict("atp", 1, 2, MatchContext())

    def test_search_finds_players(self, predictor):
        found = predictor.find("atp", "Player")
        assert not found.empty


class TestApi:
    @pytest.fixture(scope="class")
    def client(self, bundle):
        from tennisdash.api import server

        server._state["predictor"] = MatchPredictor(bundle)
        return TestClient(server.app)

    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_players(self, client):
        payload = client.get("/api/players", params={"tour": "atp", "limit": 5}).json()
        assert 0 < len(payload) <= 5
        assert {"player_id", "name"} <= set(payload[0])

    def test_rankings(self, client):
        payload = client.get("/api/rankings", params={"tour": "atp", "top": 10,
                                                      "min_matches": 1}).json()
        assert payload
        elos = [row["elo"] for row in payload]
        assert elos == sorted(elos, reverse=True)

    def test_predict_endpoint(self, client, two_players):
        a, b = two_players
        payload = client.get("/api/predict", params={
            "tour": "atp", "p1": a, "p2": b, "surface": "Clay", "best_of": 5,
        }).json()
        assert 0 < payload["p1_win_probability"] < 1
        assert payload["p1_win_probability"] + payload["p2_win_probability"] == pytest.approx(1.0)
        assert payload["context"]["surface"] == "Clay"
        assert payload["base_models"]

    def test_predict_is_json_clean(self, client, two_players):
        """NaN is not valid JSON; the response must never contain a bare NaN."""
        a, b = two_players
        raw = client.get("/api/predict", params={"tour": "atp", "p1": a, "p2": b}).text
        assert "NaN" not in raw and "Infinity" not in raw

    def test_player_profile(self, client, two_players):
        a, _ = two_players
        payload = client.get(f"/api/player/atp/{a}").json()
        assert payload["player_id"] == a
        assert "elo" in payload and "serve_return" in payload
        assert payload["elo"]["overall"] > 0

    def test_unknown_player_is_404(self, client):
        assert client.get("/api/player/atp/99999999").status_code == 404

    def test_h2h(self, client, two_players):
        a, b = two_players
        payload = client.get("/api/h2h", params={"tour": "atp", "p1": a, "p2": b,
                                                 "surface": "Hard"}).json()
        assert payload["total"] == payload["p1_wins"] + payload["p2_wins"]
        assert 0 <= payload["model_value"]["h2h_win_pct"] <= 1

    def test_model_card(self, client):
        payload = client.get("/api/model").json()
        assert payload["training_rows"] > 0
        assert payload["stacker_weights"]

    def test_dashboard_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Sports-Dash" in response.text
