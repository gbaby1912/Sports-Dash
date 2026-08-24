"""Closed-form hierarchical tennis model: point -> game -> set -> match.

Tennis has an unusually well-behaved scoring structure. If you assume points on
serve are independent and identically distributed within a match, the
probability of winning a game, a tiebreak, a set and the match all follow in
closed form from just two numbers: each player's probability of winning a point
on their own serve.

That assumption is not literally true - there is measurable point-to-point
dependence around break points and in tiebreaks - but the errors are small and
largely cancel, and the resulting model is a genuinely *different* view of a
matchup from a rating system. Elo asks "who has been winning?"; this asks "given
how these two serve and return, who should win?". Blending the two beats either
alone, which is the whole reason it is here.

Everything is vectorised over numpy arrays so a whole season can be scored at
once.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

ArrayLike = np.ndarray | float


def _as_array(value: ArrayLike) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


def game_win_probability(p: ArrayLike) -> np.ndarray:
    """P(server holds) given point-win-on-serve probability ``p``.

    Derived by summing the paths to 4-0, 4-1, 4-2 and then the deuce branch,
    where from deuce the server wins with p^2 / (p^2 + q^2).
    """
    p = np.clip(_as_array(p), 1e-6, 1 - 1e-6)
    q = 1.0 - p

    # Win to love, 15, 30 (i.e. opponent gets 0, 1, 2 points).
    to_love = p ** 4
    to_15 = 4 * (p ** 4) * q
    to_30 = 10 * (p ** 4) * (q ** 2)
    # Reach deuce (3-3) then win the deuce sub-game.
    reach_deuce = 20 * (p ** 3) * (q ** 3)
    deuce = (p ** 2) / (p ** 2 + q ** 2)
    return to_love + to_15 + to_30 + reach_deuce * deuce


def tiebreak_win_probability(p: ArrayLike, q: ArrayLike, target: int = 7) -> np.ndarray:
    """P(player A wins a tiebreak) serving first.

    ``p`` is A's point-win probability on A's serve, ``q`` is B's on B's serve.
    Solved by forward dynamic programming over (points_a, points_b, points_played)
    because the serving order alternates in pairs after the first point.
    """
    p = np.clip(_as_array(p), 1e-6, 1 - 1e-6)
    q = np.clip(_as_array(q), 1e-6, 1 - 1e-6)
    p, q = np.broadcast_arrays(p, q)
    shape = p.shape

    cap = target + 28  # beyond this the sudden-death branch takes over
    # state[(a, b)] = probability mass currently at that score
    state = {(0, 0): np.ones(shape)}
    win = np.zeros(shape)

    for _ in range(2 * cap):
        next_state: dict[tuple[int, int], np.ndarray] = {}
        for (a, b), mass in state.items():
            served = a + b
            # A serves point 1, then B serves points 2-3, A serves 4-5, ...
            a_serves = ((served + 1) // 2) % 2 == 0
            p_a_wins_point = p if a_serves else 1.0 - q

            for delta, prob in ((1, p_a_wins_point), (0, 1.0 - p_a_wins_point)):
                na, nb = (a + 1, b) if delta else (a, b + 1)
                flow = mass * prob
                if na >= target and na - nb >= 2:
                    win = win + flow
                    continue
                if nb >= target and nb - na >= 2:
                    continue
                if na >= cap or nb >= cap:
                    # From a long deuce-style tail both players are at 50/50 on
                    # the two-point margin, so split the remaining mass.
                    win = win + flow * _sudden_death(p, q, na - nb)
                    continue
                key = (na, nb)
                next_state[key] = next_state.get(key, 0.0) + flow
        state = next_state
        if not state:
            break

    return win.reshape(shape) if shape else win


def _sudden_death(p: np.ndarray, q: np.ndarray, lead: int) -> np.ndarray:
    """Exact win probability deep in a tiebreak, given the current lead.

    Past the truncation point the serving order is a clean alternation of pairs -
    over any two consecutive points each player serves once - so the lead
    performs a random walk in steps of +/-2. Writing ``a = p(1-q)`` for the
    chance A takes both points of a pair and ``b = (1-p)q`` for B taking both,
    the walk moves up with probability ``s = a / (a + b)`` conditional on moving
    at all, and pairs that split leave the lead unchanged.

    Only leads of -1, 0 and +1 can reach here: any larger lead has already been
    absorbed as a win or a loss by the caller. Solving the walk on each gives

        f(0)  = s
        f(+1) = s  / (1 - s(1 - s))
        f(-1) = s^2 / (1 - s(1 - s))

    (f(+1) and f(-1) satisfy f(+1) = s + (1-s)f(-1), f(-1) = s f(+1).)

    These are exact rather than approximate, which matters: the previous
    approximation was not antisymmetric between the two players and leaked a
    small bias into every tiebreak - enough to move an evenly-matched tiebreak
    off 0.5 in the sixth decimal place.
    """
    a = p * (1.0 - q)
    b = (1.0 - p) * q
    total = a + b
    s = np.where(total > 1e-12, a / np.maximum(total, 1e-12), 0.5)
    if lead >= 2:
        return np.ones_like(s)
    if lead <= -2:
        return np.zeros_like(s)
    if lead == 0:
        return s
    denominator = np.maximum(1.0 - s * (1.0 - s), 1e-12)
    return s / denominator if lead > 0 else (s * s) / denominator


def set_win_probability(
    p: ArrayLike,
    q: ArrayLike,
    tiebreak: bool = True,
    advantage_set: bool = False,
) -> np.ndarray:
    """P(player A wins a set), averaged over who serves first.

    Forward DP over game scores. Serving alternates every game, so the server of
    game ``n`` is determined by ``n`` and by who opened the set; we average the
    two openings because the toss is a coin flip and models that ignore this
    introduce a small but systematic bias.
    """
    p = np.clip(_as_array(p), 1e-6, 1 - 1e-6)
    q = np.clip(_as_array(q), 1e-6, 1 - 1e-6)
    p, q = np.broadcast_arrays(p, q)

    hold_a = game_win_probability(p)
    hold_b = game_win_probability(q)
    tb_a_first = tiebreak_win_probability(p, q)
    # If B serves first in the tiebreak, A's chance is the complement of B's.
    tb_b_first = 1.0 - tiebreak_win_probability(q, p)

    totals = []
    for a_serves_first in (True, False):
        state = {(0, 0): np.ones(p.shape)}
        win = np.zeros(p.shape)
        for _ in range(40):
            next_state: dict[tuple[int, int], np.ndarray] = {}
            for (ga, gb), mass in state.items():
                games = ga + gb
                a_serving = (games % 2 == 0) == a_serves_first
                p_a_wins_game = hold_a if a_serving else 1.0 - hold_b

                for a_won, prob in ((True, p_a_wins_game), (False, 1.0 - p_a_wins_game)):
                    na, nb = (ga + 1, gb) if a_won else (ga, gb + 1)
                    flow = mass * prob
                    if na >= 6 and na - nb >= 2:
                        win = win + flow
                        continue
                    if nb >= 6 and nb - na >= 2:
                        continue
                    if na == 6 and nb == 6:
                        if advantage_set:
                            # Long final set: whoever breaks first wins. The
                            # random walk over game pairs converges to this.
                            win = win + flow * _long_set(hold_a, hold_b)
                        else:
                            # The tiebreak is served by whoever is next in turn.
                            next_server_is_a = ((na + nb) % 2 == 0) == a_serves_first
                            win = win + flow * np.where(next_server_is_a, tb_a_first, tb_b_first)
                        continue
                    if na > 8 or nb > 8:
                        continue
                    key = (na, nb)
                    next_state[key] = next_state.get(key, 0.0) + flow
            state = next_state
            if not state:
                break
        totals.append(win)

    return 0.5 * (totals[0] + totals[1])


def _long_set(hold_a: np.ndarray, hold_b: np.ndarray) -> np.ndarray:
    """P(A wins an advantage set from 6-6) - first player to break, wins."""
    break_a = 1.0 - hold_b   # A breaks B
    break_b = 1.0 - hold_a   # B breaks A
    # Over each pair of games (one hold each way), A wins the pair outright with
    # hold_a*break_a and loses it with break_b*hold_b.
    win_pair = hold_a * break_a
    lose_pair = break_b * hold_b
    total = win_pair + lose_pair
    return np.where(total > 1e-9, win_pair / np.maximum(total, 1e-9), 0.5)


def match_win_probability(
    p: ArrayLike,
    q: ArrayLike,
    best_of: int = 3,
    final_set_tiebreak: bool = True,
) -> np.ndarray:
    """P(player A wins the match) from serve point-win probabilities."""
    p_set = set_win_probability(p, q)
    if final_set_tiebreak:
        p_final = p_set
    else:
        p_final = set_win_probability(p, q, advantage_set=True)

    if best_of == 3:
        # 2-0, plus 2-1 through either order of the first two sets.
        two_nil = p_set ** 2
        two_one = 2 * p_set * (1 - p_set) * p_final
        return two_nil + two_one
    if best_of == 5:
        s, f = p_set, p_final
        three_nil = s ** 3
        three_one = 3 * (s ** 3) * (1 - s)
        # 3-2 requires the fifth set, which may use different rules.
        three_two = 6 * (s ** 2) * ((1 - s) ** 2) * f
        return three_nil + three_one + three_two
    raise ValueError("best_of must be 3 or 5")


def score_distribution(
    p: ArrayLike,
    q: ArrayLike,
    best_of: int = 3,
) -> dict[str, float]:
    """Probability of each possible set scoreline, from A's perspective."""
    p_set = float(np.mean(set_win_probability(p, q)))
    s = p_set
    if best_of == 3:
        return {
            "2-0": s ** 2,
            "2-1": 2 * s * (1 - s) * s,
            "1-2": 2 * s * (1 - s) * (1 - s),
            "0-2": (1 - s) ** 2,
        }
    return {
        "3-0": s ** 3,
        "3-1": 3 * (s ** 3) * (1 - s),
        "3-2": 6 * (s ** 2) * ((1 - s) ** 2) * s,
        "2-3": 6 * (s ** 2) * ((1 - s) ** 2) * (1 - s),
        "1-3": 3 * (1 - s) ** 3 * s,
        "0-3": (1 - s) ** 3,
    }


@lru_cache(maxsize=4096)
def _cached_match_prob(p: float, q: float, best_of: int) -> float:
    return float(match_win_probability(p, q, best_of=best_of)[0])


def match_probability_scalar(p: float, q: float, best_of: int = 3) -> float:
    """Cached scalar entry point for interactive/one-off use."""
    return _cached_match_prob(round(float(p), 4), round(float(q), 4), int(best_of))
