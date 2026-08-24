"""FastAPI service backing the dashboard.

The bundle is loaded once at startup and held in memory - it contains the fitted
engines, so predictions are a feature assembly plus four model evaluations, on
the order of a few milliseconds.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT
from ..predict import MatchContext, MatchPredictor
from ..train import load_bundle

log = logging.getLogger(__name__)

app = FastAPI(
    title="Sports-Dash Tennis Predictor",
    description="Calibrated ATP/WTA match prediction",
    version="1.0.0",
)

WEB_DIR = ROOT / "web"
_state: dict = {}


def get_predictor() -> MatchPredictor:
    if "predictor" not in _state:
        try:
            _state["predictor"] = MatchPredictor(load_bundle())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _state["predictor"]


@app.on_event("startup")
def _warm_start() -> None:
    try:
        get_predictor()
        log.info("model bundle loaded")
    except HTTPException:
        log.warning("no trained model found; API will report 503 until one is trained")


def _clean(value):
    """JSON-safe conversion: NaN and numpy scalars are not valid JSON."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int)):
        return value
    if pd.isna(value):
        return None
    return str(value)


@app.get("/api/health")
def health() -> dict:
    ready = "predictor" in _state
    return {"status": "ok" if ready else "no-model", "model_loaded": ready}


@app.get("/api/model")
def model_card() -> JSONResponse:
    """Everything the dashboard needs to describe the model honestly."""
    predictor = get_predictor()
    meta = dict(predictor.metadata)
    return JSONResponse(_clean(meta))


@app.get("/api/players")
def players(
    tour: str = Query("atp"),
    q: str = Query("", description="name substring"),
    limit: int = Query(30, le=200),
    min_matches: int = Query(0),
) -> JSONResponse:
    predictor = get_predictor()
    block = predictor.directory[predictor.directory["tour"] == tour.lower()]
    block = block[block["matches"] >= min_matches]
    if q:
        block = block[block["name"].str.contains(q, case=False, na=False, regex=False)]
    block = block.sort_values(["last_played", "matches"], ascending=[False, False]).head(limit)
    records = block[[
        "player_id", "name", "hand", "height_cm", "ioc",
        "last_rank", "matches", "win_pct", "last_played",
    ]].to_dict("records")
    return JSONResponse(_clean(records))


@app.get("/api/rankings")
def rankings(
    tour: str = Query("atp"),
    surface: str | None = Query(None),
    top: int = Query(30, le=200),
    min_matches: int = Query(15),
    active_days: int = Query(
        365, description="Only rank players who competed within this many days "
                         "of the end of the data. 0 disables the filter."
    ),
) -> JSONResponse:
    predictor = get_predictor()
    active_since = None
    if active_days > 0:
        span = (predictor.metadata.get("data_span") or [None, None])[1]
        if span:
            active_since = pd.Timestamp(span) - pd.Timedelta(days=active_days)
    snapshot = predictor.elo.snapshot(tour.lower(), surface=surface, active_since=active_since)
    if snapshot.empty:
        return JSONResponse([])
    snapshot = snapshot[snapshot["matches"] >= min_matches].head(top)
    names = predictor.directory[predictor.directory.tour == tour.lower()].set_index("player_id")
    snapshot = snapshot.copy()
    snapshot["name"] = snapshot["player_id"].map(names["name"])
    snapshot["ioc"] = snapshot["player_id"].map(names["ioc"])

    fit = predictor.rolling.latest
    if fit is not None:
        # None rather than 0.0 for a player the fit has never seen - see
        # ServeReturnFit.has_rating.
        snapshot["serve_skill"] = [
            fit.serve_skill(tour.lower(), int(p), surface)
            if fit.has_rating(tour.lower(), int(p)) else None
            for p in snapshot["player_id"]
        ]
        snapshot["return_skill"] = [
            fit.return_skill(tour.lower(), int(p), surface)
            if fit.has_rating(tour.lower(), int(p)) else None
            for p in snapshot["player_id"]
        ]
    return JSONResponse(_clean(snapshot.to_dict("records")))


@app.get("/api/player/{tour}/{player_id}")
def player_profile(tour: str, player_id: int, surface: str | None = None) -> JSONResponse:
    """Full profile: ratings, adjusted serve/return, form, fatigue and clutch."""
    predictor = get_predictor()
    tour = tour.lower()
    try:
        info = predictor.player(tour, player_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    surfaces = ["Hard", "Clay", "Grass"]
    fit = predictor.rolling.latest
    profile: dict = {
        "player_id": int(player_id),
        "tour": tour,
        "name": info.name,
        "hand": info.hand,
        "height_cm": info.height_cm,
        "ioc": info.ioc,
        "rank": info.last_rank,
        "matches": int(info.matches),
        "win_pct": float(info.win_pct),
        "last_played": info.last_played,
        "elo": {
            "overall": predictor.elo.rating_of(tour, player_id, "overall"),
            "points": predictor.elo.rating_of(tour, player_id, "points"),
            "games": predictor.elo.rating_of(tour, player_id, "games"),
            "peak": predictor.elo.peak(tour, player_id),
            "matches": predictor.elo.matches_played(tour, player_id),
            **{s.lower(): predictor.elo.blended(tour, player_id, s) for s in surfaces},
        },
    }

    if fit is not None:
        profile["serve_return"] = {
            "overall": {
                "serve_skill": fit.serve_skill(tour, player_id),
                "return_skill": fit.return_skill(tour, player_id),
                "raw_spw": fit.raw_spw.get((tour, player_id)),
                "raw_rpw": fit.raw_rpw.get((tour, player_id)),
                "service_points": fit.coverage(tour, player_id),
            },
            "rated": fit.has_rating(tour, player_id),
            "by_surface": {
                s.lower(): {
                    "serve_skill": fit.serve_skill(tour, player_id, s),
                    "return_skill": fit.return_skill(tour, player_id, s),
                    "neutral_spw": fit.spw_vs_average_returner(tour, player_id, s),
                    "neutral_rpw": fit.rpw_vs_average_server(tour, player_id, s),
                }
                for s in surfaces
            },
        }

    state = predictor.history.players.get((tour, player_id))
    if state is not None:
        # Anchor the rolling windows to the end of the data, not to the wall
        # clock. With a historical archive "days since last match" measured from
        # today is technically true and completely useless - it just reports how
        # stale the dataset is.
        span = (predictor.metadata.get("data_span") or [None, None])[1]
        as_of = pd.Timestamp(span) if span else pd.Timestamp.today().normalize()
        profile["as_of"] = as_of
        profile["form"] = _clean(predictor.history._form(state, as_of))
        profile["fatigue"] = _clean(predictor.history._fatigue(state, as_of))
        profile["clutch"] = _clean(predictor.history._clutch(state, tour))
        profile["record"] = {
            "career_matches": state.matches,
            "career_wins": state.wins,
            "win_streak": state.win_streak,
            "loss_streak": state.loss_streak,
            "rated": fit.has_rating(tour, player_id),
            "by_surface": {
                s.lower(): {
                    "matches": state.surface_matches.get(s, 0.0),
                    "wins": state.surface_wins.get(s, 0.0),
                }
                for s in surfaces
            },
        }
    return JSONResponse(_clean(profile))


@app.get("/api/predict")
def predict(
    tour: str = Query("atp"),
    p1: int = Query(...),
    p2: int = Query(...),
    surface: str = Query("Hard"),
    best_of: int = Query(3),
    level: str = Query("A"),
    round: str = Query("R32"),
    tournament: str = Query("Neutral Court"),
    indoor: bool | None = Query(None),
    altitude_m: int | None = Query(None),
) -> JSONResponse:
    predictor = get_predictor()
    context = MatchContext(
        surface=surface,
        best_of=best_of,
        level=level,
        round=round,
        tourney_name=tournament,
        indoor=indoor,
        altitude_m=altitude_m,
    )
    try:
        prediction = predictor.predict(tour.lower(), p1, p2, context)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = prediction.to_dict()
    payload["context"] = {
        "surface": surface,
        "best_of": best_of,
        "indoor": context.resolved_indoor(),
        "altitude_m": context.resolved_altitude(),
        "level": level,
        "round": round,
        "tournament": tournament,
    }
    return JSONResponse(_clean(payload))


@app.get("/api/h2h")
def head_to_head(tour: str, p1: int, p2: int, surface: str | None = None) -> JSONResponse:
    """Raw head-to-head record, and the shrunk value the model actually uses."""
    predictor = get_predictor()
    tour = tour.lower()
    wins, losses = predictor.history.h2h.get((tour, p1, p2), [0.0, 0.0])
    payload = {"p1_wins": wins, "p2_wins": losses, "total": wins + losses}
    if surface:
        s_wins, s_losses = predictor.history.h2h_surface.get((tour, p1, p2, surface), [0.0, 0.0])
        payload["surface"] = {"p1_wins": s_wins, "p2_wins": s_losses, "surface": surface}
    payload["model_value"] = predictor.history._h2h(tour, p1, p2, surface or "Hard")
    return JSONResponse(_clean(payload))


# The dashboard is a static bundle served from the same origin, so there is no
# CORS configuration to get wrong.
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
