import pandas as pd
import pytest

from tennisdash.data.ingest import normalise_tour
from tennisdash.data.synthetic import SyntheticTour


@pytest.fixture(scope="session")
def small_matches() -> pd.DataFrame:
    """A small but complete normalised match table, built the real way.

    Generated through the synthetic tour and the production ingest path, so the
    fixture exercises the same code the pipeline uses rather than a hand-written
    stub that could drift from it.
    """
    frames = []
    for tour in ("atp", "wta"):
        world = SyntheticTour(tour, 2018, 2021, n_players=90, seed=99)
        frames.append(normalise_tour(world.generate(), tour))
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        ["match_date", "tour", "tourney_id", "match_num"], kind="mergesort"
    ).reset_index(drop=True)


@pytest.fixture(scope="session")
def small_features(small_matches):
    from tennisdash.features.builder import build_features

    features, engines = build_features(small_matches, save=False, refit_days=90)
    return features, engines
