"""Point-by-point match simulator.

Used in two places:

1. To generate a realistic offline dataset in the *exact* raw archive format,
   so the whole pipeline (ingest -> features -> train -> backtest -> API) can be
   exercised without network access.
2. To produce score distributions for a predicted matchup in the dashboard.

The simulator tracks the same counting stats the real archives record (service
points, first serves in/won, second-serve points won, aces, double faults,
break points faced/saved), so simulated data is consumable by the identical
ingestion code path as real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ServeProfile:
    """Everything needed to simulate one player's service points."""

    spw: float                  # probability of winning a point on serve
    first_in: float = 0.61      # first-serve in percentage
    ace_rate: float = 0.07      # aces per service point
    df_rate: float = 0.04       # double faults per service point

    def clipped(self) -> "ServeProfile":
        return ServeProfile(
            spw=float(np.clip(self.spw, 0.35, 0.88)),
            first_in=float(np.clip(self.first_in, 0.40, 0.80)),
            ace_rate=float(np.clip(self.ace_rate, 0.0, 0.35)),
            df_rate=float(np.clip(self.df_rate, 0.0, 0.20)),
        )


@dataclass
class PlayerMatchStats:
    ace: int = 0
    df: int = 0
    svpt: int = 0
    first_in: int = 0
    first_won: int = 0
    second_won: int = 0
    sv_gms: int = 0
    bp_saved: int = 0
    bp_faced: int = 0
    games_won: int = 0
    sets_won: int = 0
    tiebreaks_won: int = 0


@dataclass
class SimulatedMatch:
    winner: int                       # 0 or 1
    score: str
    sets: list[tuple[int, int]]
    tiebreak_scores: dict[int, tuple[int, int]] = field(default_factory=dict)
    stats: tuple[PlayerMatchStats, PlayerMatchStats] = field(
        default_factory=lambda: (PlayerMatchStats(), PlayerMatchStats())
    )
    minutes: int = 0


def _serve_point(
    profile: ServeProfile,
    stats: PlayerMatchStats,
    rng: np.random.Generator,
    is_break_point: bool,
) -> bool:
    """Simulate one service point, updating counting stats. True = server wins.

    The first/second serve split is reconstructed from the aggregate serve
    percentage: with first-serve-in rate ``f`` and a second-serve win rate that
    sits ~20 points below the first-serve win rate (the long-run tour gap),
    ``spw = f*w1 + (1-f)*w2`` pins down both.
    """
    stats.svpt += 1
    if is_break_point:
        stats.bp_faced += 1

    w1 = profile.spw + 0.20 * (1.0 - profile.first_in)
    w1 = float(np.clip(w1, 0.35, 0.97))
    w2 = float(np.clip(w1 - 0.20, 0.20, 0.95))

    if rng.random() < profile.first_in:
        stats.first_in += 1
        # An ace is a subset of first-serve points won.
        ace_given_first = min(profile.ace_rate / max(profile.first_in, 1e-6), w1)
        if rng.random() < ace_given_first:
            stats.ace += 1
            stats.first_won += 1
            won = True
        else:
            won = rng.random() < (w1 - ace_given_first) / max(1.0 - ace_given_first, 1e-6)
            stats.first_won += int(won)
    else:
        df_given_second = min(profile.df_rate / max(1.0 - profile.first_in, 1e-6), 0.60)
        if rng.random() < df_given_second:
            stats.df += 1
            won = False
        else:
            won = rng.random() < w2 / max(1.0 - df_given_second, 1e-6)
            stats.second_won += int(won)

    if is_break_point and won:
        stats.bp_saved += 1
    return won


def _play_game(
    profile: ServeProfile,
    stats: PlayerMatchStats,
    rng: np.random.Generator,
) -> bool:
    """Simulate a service game. True = server holds."""
    stats.sv_gms += 1
    server_points = 0
    returner_points = 0
    while True:
        is_bp = returner_points >= 3 and returner_points >= server_points + 1
        if _serve_point(profile, stats, rng, is_bp):
            server_points += 1
        else:
            returner_points += 1
        if server_points >= 4 and server_points - returner_points >= 2:
            return True
        if returner_points >= 4 and returner_points - server_points >= 2:
            return False


def _play_tiebreak(
    profiles: tuple[ServeProfile, ServeProfile],
    stats: tuple[PlayerMatchStats, PlayerMatchStats],
    server: int,
    rng: np.random.Generator,
    target: int = 7,
) -> tuple[int, int, int]:
    """Simulate a tiebreak. Returns (winner, points_a, points_b)."""
    points = [0, 0]
    current = server
    served = 0
    while True:
        # Tiebreak serving order: 1 point, then alternating pairs.
        won = _serve_point(profiles[current], stats[current], rng, is_break_point=False)
        points[current if won else 1 - current] += 1
        served += 1
        if served == 1 or (served - 1) % 2 == 0:
            current = 1 - current
        if max(points) >= target and abs(points[0] - points[1]) >= 2:
            break
        if max(points) >= target + 12:  # safety valve
            break
    winner = 0 if points[0] > points[1] else 1
    return winner, points[0], points[1]


def simulate_match(
    profile_a: ServeProfile,
    profile_b: ServeProfile,
    best_of: int = 3,
    rng: np.random.Generator | None = None,
    final_set_tiebreak: bool = True,
) -> SimulatedMatch:
    """Simulate a full match point by point and return score plus stats."""
    rng = rng or np.random.default_rng()
    profiles = (profile_a.clipped(), profile_b.clipped())
    stats = (PlayerMatchStats(), PlayerMatchStats())
    sets_won = [0, 0]
    set_scores: list[tuple[int, int]] = []
    tiebreak_scores: dict[int, tuple[int, int]] = {}
    sets_to_win = best_of // 2 + 1
    server = int(rng.random() < 0.5)
    total_games = 0

    while max(sets_won) < sets_to_win:
        games = [0, 0]
        while True:
            if games[0] == 6 and games[1] == 6:
                is_final_set = sum(sets_won) == best_of - 1
                target = 10 if (is_final_set and not final_set_tiebreak) else 7
                tb_winner, pa, pb = _play_tiebreak(profiles, stats, server, rng, target=target)
                games[tb_winner] += 1
                stats[tb_winner].tiebreaks_won += 1
                tiebreak_scores[len(set_scores)] = (pa, pb)
                break

            held = _play_game(profiles[server], stats[server], rng)
            winner_of_game = server if held else 1 - server
            games[winner_of_game] += 1
            stats[winner_of_game].games_won += 1
            total_games += 1
            server = 1 - server

            if max(games) >= 6 and abs(games[0] - games[1]) >= 2:
                break

        set_winner = 0 if games[0] > games[1] else 1
        sets_won[set_winner] += 1
        stats[set_winner].sets_won += 1
        set_scores.append((games[0], games[1]))

    winner = 0 if sets_won[0] > sets_won[1] else 1
    parts = []
    for index, (ga, gb) in enumerate(set_scores):
        # Scorelines in the archives are always written winner-first.
        first, second = (ga, gb) if winner == 0 else (gb, ga)
        piece = f"{first}-{second}"
        if index in tiebreak_scores:
            pa, pb = tiebreak_scores[index]
            loser_points = min(pa, pb)
            piece += f"({loser_points})"
        parts.append(piece)

    # Roughly 5.5 minutes a game plus a fixed changeover overhead.
    minutes = int(total_games * 5.6 + len(set_scores) * 4 + rng.normal(0, 6))
    return SimulatedMatch(
        winner=winner,
        score=" ".join(parts),
        sets=set_scores,
        tiebreak_scores=tiebreak_scores,
        stats=stats,
        minutes=max(minutes, 20),
    )
