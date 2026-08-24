"""The point-to-match model is pure maths, so it can be tested exactly."""
import numpy as np
import pytest

from tennisdash.data.simulate import ServeProfile, simulate_match
from tennisdash.models.markov import (
    game_win_probability,
    match_win_probability,
    score_distribution,
    set_win_probability,
    tiebreak_win_probability,
)


@pytest.mark.parametrize(
    "p,expected",
    [(0.50, 0.5000), (0.55, 0.6230), (0.60, 0.7357), (0.65, 0.8299), (0.70, 0.9008)],
)
def test_game_hold_matches_closed_form(p, expected):
    assert float(game_win_probability(p)[0]) == pytest.approx(expected, abs=5e-4)


def test_equal_servers_are_even_everywhere():
    for p in (0.55, 0.62, 0.70):
        assert float(game_win_probability(p)[0]) == pytest.approx(
            1 - float(game_win_probability(1 - p)[0]), abs=1e-9
        )
        assert float(tiebreak_win_probability(p, p)[0]) == pytest.approx(0.5, abs=1e-6)
        assert float(set_win_probability(p, p)[0]) == pytest.approx(0.5, abs=1e-6)
        for best_of in (3, 5):
            assert float(match_win_probability(p, p, best_of=best_of)[0]) == pytest.approx(
                0.5, abs=1e-6
            )


def test_match_probability_is_exactly_antisymmetric():
    for p, q, best_of in [(0.66, 0.61, 3), (0.70, 0.58, 5), (0.59, 0.64, 3)]:
        forward = float(match_win_probability(p, q, best_of=best_of)[0])
        backward = float(match_win_probability(q, p, best_of=best_of)[0])
        assert forward + backward == pytest.approx(1.0, abs=1e-12)


def test_best_of_five_amplifies_the_favourite():
    """More sets means less variance, so the better player wins more often."""
    bo3 = float(match_win_probability(0.66, 0.61, best_of=3)[0])
    bo5 = float(match_win_probability(0.66, 0.61, best_of=5)[0])
    assert bo5 > bo3 > 0.5


def test_probability_is_monotone_in_serve_strength():
    previous = 0.0
    for p in np.arange(0.55, 0.76, 0.02):
        current = float(match_win_probability(p, 0.62, best_of=3)[0])
        assert current > previous
        previous = current


def test_score_distribution_sums_to_one():
    for best_of in (3, 5):
        distribution = score_distribution(0.67, 0.60, best_of=best_of)
        assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(v >= 0 for v in distribution.values())


@pytest.mark.parametrize("p,q,best_of", [(0.66, 0.60, 3), (0.68, 0.60, 5), (0.58, 0.63, 3)])
def test_closed_form_agrees_with_point_by_point_simulation(p, q, best_of):
    """The strongest check available: two independent implementations must agree.

    The closed form sums over scoring paths; the simulator plays out points one
    at a time. They share no code, so agreement inside Monte Carlo error means
    the recursion is right.
    """
    rng = np.random.default_rng(20240101)
    n = 6000
    wins = sum(
        simulate_match(ServeProfile(spw=p), ServeProfile(spw=q), best_of=best_of, rng=rng).winner == 0
        for _ in range(n)
    )
    simulated = wins / n
    closed = float(match_win_probability(p, q, best_of=best_of)[0])
    standard_error = (simulated * (1 - simulated) / n) ** 0.5
    assert abs(closed - simulated) < 4 * standard_error


def test_vectorised_and_scalar_paths_agree():
    p = np.array([0.60, 0.65, 0.70])
    q = np.array([0.62, 0.62, 0.62])
    batch = match_win_probability(p, q, best_of=3)
    for i in range(3):
        assert batch[i] == pytest.approx(
            float(match_win_probability(p[i], q[i], best_of=3)[0]), abs=1e-12
        )
