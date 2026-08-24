"""Generate a realistic synthetic tour in the *raw archive CSV format*.

This exists so the entire pipeline can be built, tested and demonstrated
without network access to the public archives. Crucially it writes the same
columns the real archives use, so the data flows through the identical
ingestion, feature and training code paths - the offline path is not a
shortcut around the real one.

The generated world has genuine latent structure for the model to find:

* per-player serve and return skill, with per-surface deviations
* skill that drifts season to season (a random walk) plus an age curve, so
  "current form" is a real, learnable signal rather than noise
* a fatigue effect from recent match load
* height driving ace rate, handedness giving lefties a small edge vs righties
* a seasonal calendar (hard -> clay -> grass -> hard -> indoor) and seeded draws

It is a simulation, not real data. Numbers produced from it describe the model's
behaviour on this world, and are reported as such.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import RAW_DIR
from .simulate import ServeProfile, simulate_match

log = logging.getLogger(__name__)

# Tour-level baselines: probability of winning a point on serve, by surface.
# These are close to the real long-run averages for each tour.
BASE_SPW = {
    "atp": {"Hard": 0.646, "Clay": 0.628, "Grass": 0.664, "Carpet": 0.660},
    "wta": {"Hard": 0.585, "Clay": 0.573, "Grass": 0.597, "Carpet": 0.594},
}

COUNTRIES = [
    "USA", "ESP", "FRA", "ARG", "GER", "ITA", "AUS", "RUS", "SRB", "SUI",
    "GBR", "CZE", "CRO", "JPN", "CAN", "BEL", "NED", "POL", "AUT", "BRA",
    "CHN", "ROU", "SVK", "SWE", "UKR", "KAZ", "GRE", "NOR", "DEN", "TUN",
]

# A season's shape: (week, name, surface, level, draw_size, indoor)
def _calendar(tour: str) -> list[tuple[int, str, str, str, int, bool]]:
    events: list[tuple[int, str, str, str, int, bool]] = [
        (1, "Brisbane", "Hard", "A", 32, False),
        (2, "Auckland", "Hard", "A", 32, False),
        (3, "Australian Open", "Hard", "G", 128, False),
        (6, "Rotterdam", "Hard", "A", 32, True),
        (7, "Rio de Janeiro", "Clay", "A", 32, False),
        (8, "Dubai", "Hard", "A", 32, False),
        (10, "Indian Wells Masters", "Hard", "M", 96, False),
        (12, "Miami Masters", "Hard", "M", 96, False),
        (15, "Monte Carlo Masters", "Clay", "M", 56, False),
        (16, "Barcelona", "Clay", "A", 48, False),
        (17, "Madrid Masters", "Clay", "M", 56, False),
        (19, "Rome Masters", "Clay", "M", 56, False),
        (20, "Bogota", "Clay", "A", 32, False),
        (22, "Roland Garros", "Clay", "G", 128, False),
        (25, "Halle", "Grass", "A", 32, False),
        (26, "Queens Club", "Grass", "A", 32, False),
        (27, "Wimbledon", "Grass", "G", 128, False),
        (30, "Gstaad", "Clay", "A", 32, False),
        (31, "Kitzbuhel", "Clay", "A", 32, False),
        (32, "Washington", "Hard", "A", 48, False),
        (33, "Canada Masters", "Hard", "M", 56, False),
        (34, "Cincinnati Masters", "Hard", "M", 56, False),
        (36, "US Open", "Hard", "G", 128, False),
        (40, "Tokyo", "Hard", "A", 32, False),
        (41, "Shanghai Masters", "Hard", "M", 56, False),
        (43, "Vienna", "Hard", "A", 32, True),
        (44, "Basel", "Hard", "A", 32, True),
        (45, "Paris Masters", "Hard", "M", 48, True),
        (47, "Tour Finals", "Hard", "F", 8, True),
    ]
    if tour == "wta":
        renames = {
            "Queens Club": "Birmingham",
            "Vienna": "Linz",
            "Basel": "Luxembourg",
            "Paris Masters": "Moscow",
            "Tour Finals": "WTA Finals",
            "Monte Carlo Masters": "Stuttgart",
        }
        events = [
            (w, renames.get(n, n), "Clay" if renames.get(n) == "Stuttgart" else s,
             lvl, d, True if renames.get(n) in {"Linz", "Luxembourg", "Moscow", "Stuttgart"} else ind)
            for (w, n, s, lvl, d, ind) in events
        ]
    return events


class SyntheticTour:
    """A latent-skill world that produces archive-format match records."""

    def __init__(
        self,
        tour: str,
        start_year: int,
        end_year: int,
        n_players: int = 300,
        seed: int = 20240101,
    ) -> None:
        self.tour = tour
        self.start_year = start_year
        self.end_year = end_year
        self.rng = np.random.default_rng(seed + (0 if tour == "atp" else 991))
        self.n_players = n_players
        self._build_players()
        # Rolling ranking points, decayed weekly.
        self.points = np.zeros(n_players)
        self.recent_load: dict[int, list[tuple[pd.Timestamp, int]]] = {}

    # ------------------------------------------------------------------ setup
    def _build_players(self) -> None:
        rng = self.rng
        n = self.n_players
        # Skill is expressed in "points won" units relative to tour average.
        self.serve_skill = rng.normal(0, 0.032, n)
        self.return_skill = rng.normal(0, 0.030, n)
        # Serve and return ability are mildly negatively correlated in reality
        # (big servers tend to be weaker returners), which creates genuine
        # style matchups rather than a single dominance axis.
        self.return_skill -= 0.22 * self.serve_skill
        # A latent "overall class" term keeps the two loosely tied together.
        overall = rng.normal(0, 0.024, n)
        self.serve_skill += overall
        self.return_skill += overall

        self.surface_serve = {
            surface: rng.normal(0, 0.011, n) for surface in ("Hard", "Clay", "Grass", "Carpet")
        }
        self.surface_return = {
            surface: rng.normal(0, 0.011, n) for surface in ("Hard", "Clay", "Grass", "Carpet")
        }
        mean_height = 185 if self.tour == "atp" else 174
        self.height = np.clip(rng.normal(mean_height, 7, n), 155, 211).round()
        self.hand = np.where(rng.random(n) < 0.13, "L", "R")
        self.ioc = rng.choice(COUNTRIES, n)
        # Careers: a debut year and a career length.
        span = self.end_year - self.start_year
        self.debut_year = self.start_year - rng.integers(0, 9, n) + rng.integers(0, span + 4, n)
        self.debut_year = np.clip(self.debut_year, self.start_year - 8, self.end_year)
        self.career_years = rng.integers(6, 17, n)
        self.debut_age = rng.uniform(17.0, 21.5, n)
        self.peak_age = rng.normal(26.0, 2.0, n)
        self.player_id = 100000 + np.arange(n)
        self.name = [f"{self.tour.upper()} Player {i:03d}" for i in range(n)]
        # Season-to-season drift, drawn lazily.
        self._drift_cache: dict[int, np.ndarray] = {}

    def _drift(self, year: int) -> np.ndarray:
        """Cumulative random-walk skill drift up to ``year``."""
        if year not in self._drift_cache:
            previous = self._drift_cache.get(year - 1)
            if previous is None:
                previous = np.zeros(self.n_players)
            step = self.rng.normal(0, 0.011, self.n_players)
            self._drift_cache[year] = previous * 0.88 + step
        return self._drift_cache[year]

    def _age(self, index: np.ndarray | int, year: int) -> np.ndarray:
        return self.debut_age[index] + (year - self.debut_year[index])

    def _age_curve(self, age) -> np.ndarray:
        """Inverted-U ageing effect on skill, in points-won units."""
        peak = self.peak_age if np.ndim(age) else self.peak_age.mean()
        return -0.00055 * (np.asarray(age) - peak) ** 2

    def active_players(self, year: int) -> np.ndarray:
        alive = (year >= self.debut_year) & (year < self.debut_year + self.career_years)
        return np.flatnonzero(alive)

    # ------------------------------------------------------------- simulation
    def _fatigue(self, player: int, date: pd.Timestamp) -> float:
        """Skill penalty from matches played in the previous 14 days."""
        history = self.recent_load.get(player)
        if not history:
            return 0.0
        cutoff = date - pd.Timedelta(days=14)
        minutes = sum(m for d, m in history if d >= cutoff)
        # ~0.9 points-won penalty per 1000 minutes of recent court time.
        return -0.0009 * (minutes / 100.0)

    def _profile(
        self,
        player: int,
        opponent: int,
        surface: str,
        indoor: bool,
        altitude: int,
        year: int,
        date: pd.Timestamp,
    ) -> ServeProfile:
        drift = self._drift(year)
        age = self._age(player, year)
        serve = (
            self.serve_skill[player]
            + self.surface_serve[surface][player]
            + drift[player]
            + self._age_curve(age)
            + self._fatigue(player, date)
        )
        ret = (
            self.return_skill[opponent]
            + self.surface_return[surface][opponent]
            + drift[opponent]
            + self._age_curve(self._age(opponent, year))
            + self._fatigue(opponent, date)
        )
        base = BASE_SPW[self.tour][surface]
        spw = base + serve - ret
        # Indoor courts and thin air both favour the server.
        if indoor:
            spw += 0.009
        spw += 0.000012 * max(altitude - 500, 0)
        # Left-handers gain a small edge against right-handers.
        if self.hand[player] == "L" and self.hand[opponent] == "R":
            spw += 0.006
        # Height drives ace rate, which in turn nudges serve percentage.
        height_z = (self.height[player] - self.height.mean()) / max(self.height.std(), 1e-6)
        ace_rate = float(np.clip(0.075 + 0.028 * height_z + 1.1 * self.serve_skill[player], 0.01, 0.30))
        if surface == "Grass":
            ace_rate *= 1.25
        elif surface == "Clay":
            ace_rate *= 0.72
        return ServeProfile(
            spw=float(spw),
            first_in=float(np.clip(0.615 - 0.35 * self.serve_skill[player], 0.50, 0.72)),
            ace_rate=ace_rate,
            df_rate=float(np.clip(0.042 - 0.15 * self.serve_skill[player], 0.012, 0.10)),
        )

    def _seeded_field(self, year: int, draw_size: int) -> np.ndarray:
        """Pick a field: strongest players enter, with some randomness."""
        active = self.active_players(year)
        if len(active) == 0:
            return active
        strength = self.points[active] + self.rng.normal(0, 240, len(active))
        order = active[np.argsort(-strength)]
        size = min(draw_size, len(order))
        # A tail of the field is drawn from outside the top group (qualifiers).
        top = max(int(size * 0.8), 1)
        chosen = list(order[:top])
        rest = [p for p in order[top:] if p not in set(chosen)]
        if rest and size > top:
            extra = self.rng.choice(rest, size=min(size - top, len(rest)), replace=False)
            chosen.extend(np.atleast_1d(extra).tolist())
        return np.array(chosen[:size], dtype=int)

    def _run_event(
        self,
        year: int,
        week: int,
        name: str,
        surface: str,
        level: str,
        draw_size: int,
        indoor: bool,
        rank_lookup: dict[int, tuple[int, int]],
    ) -> list[dict]:
        from .venues import venue_altitude

        field = self._seeded_field(year, draw_size)
        if len(field) < 4:
            return []
        # Round the field down to a power of two for a clean knockout bracket.
        size = 2 ** int(np.floor(np.log2(len(field))))
        field = field[:size]
        altitude = venue_altitude(name)
        best_of = 5 if (level == "G" and self.tour == "atp") else 3
        start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(weeks=week - 1)
        tourney_id = f"{year}-{abs(hash(name)) % 9000 + 1000}"

        # Seeding: the bracket is arranged so the strongest meet late.
        seeds = {int(p): i + 1 for i, p in enumerate(field[: max(size // 4, 1)])}
        bracket = list(field)
        # Standard snake seeding keeps the top two apart until the final.
        bracket = _snake_bracket(bracket)

        rows: list[dict] = []
        round_names = _round_names(size)
        match_num = 1
        alive = bracket
        # Depth reached, keyed by internal player index (not archive id).
        depth_reached: dict[int, int] = {int(p): 0 for p in field}
        for round_index, round_name in enumerate(round_names):
            next_alive = []
            day_offset = min(round_index, 13)
            match_date = start + pd.Timedelta(days=int(day_offset * (1.4 if size >= 96 else 1.0)))
            for i in range(0, len(alive), 2):
                a, b = int(alive[i]), int(alive[i + 1])
                pa = self._profile(a, b, surface, indoor, altitude, year, match_date)
                pb = self._profile(b, a, surface, indoor, altitude, year, match_date)
                result = simulate_match(pa, pb, best_of=best_of, rng=self.rng)
                winner, loser = (a, b) if result.winner == 0 else (b, a)
                w_stats = result.stats[0 if result.winner == 0 else 1]
                l_stats = result.stats[1 if result.winner == 0 else 0]

                retired = self.rng.random() < 0.018
                score = result.score
                if retired:
                    score = " ".join(result.score.split()[:1]) + " RET"

                rows.append(
                    _archive_row(
                        self, year, tourney_id, name, surface, draw_size, level,
                        start, match_num, round_name, best_of, result.minutes,
                        score, winner, loser, w_stats, l_stats, seeds, rank_lookup,
                    )
                )
                for player, stats_minutes in ((a, result.minutes), (b, result.minutes)):
                    self.recent_load.setdefault(player, []).append((match_date, stats_minutes))
                next_alive.append(winner)
                depth_reached[winner] = round_index + 1
                match_num += 1
            alive = next_alive

        # Award ranking points by round reached.
        base_points = {"G": 2000, "M": 1000, "F": 1500, "A": 250}.get(level, 250)
        total_rounds = len(round_names)
        for player, depth in depth_reached.items():
            self.points[player] += base_points * (0.5 ** (total_rounds - depth))
        return rows

    def generate(self) -> pd.DataFrame:
        """Run every season and return one archive-format frame."""
        all_rows: list[dict] = []
        for year in range(self.start_year, self.end_year + 1):
            # Ranking points decay across the season boundary (52-week rolling).
            self.points *= 0.55
            rank_lookup = _rank_lookup(self.points)
            for (week, name, surface, level, draw_size, indoor) in _calendar(self.tour):
                all_rows.extend(
                    self._run_event(
                        year, week, name, surface, level, draw_size, indoor, rank_lookup
                    )
                )
            # Drop stale fatigue history to keep memory flat.
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            for player, history in self.recent_load.items():
                self.recent_load[player] = [h for h in history if h[0] >= cutoff]
            log.info("%s %d: %d matches", self.tour, year, len(all_rows))
        return pd.DataFrame(all_rows)


def _seed_positions(size: int) -> list[int]:
    """Standard knockout seeding order (1-indexed seeds) for a draw of ``size``.

    Produces the bracket where seed 1 and seed 2 can only meet in the final,
    1 and 4 only in the semis, and so on - the same shape a real draw uses.
    """
    order = [1]
    while len(order) < size:
        pairs = len(order) * 2
        expanded: list[int] = []
        for seed in order:
            expanded.append(seed)
            expanded.append(pairs + 1 - seed)
        order = expanded
    return order


def _snake_bracket(players: list[int]) -> list[int]:
    """Arrange a strength-ordered field into a properly seeded bracket."""
    size = len(players)
    return [players[position - 1] for position in _seed_positions(size)]


def _round_names(size: int) -> list[str]:
    mapping = {128: "R128", 64: "R64", 32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "F"}
    names = []
    current = size
    while current >= 2:
        names.append(mapping.get(current, f"R{current}"))
        current //= 2
    return names


def _rank_lookup(points: np.ndarray) -> dict[int, tuple[int, int]]:
    order = np.argsort(-points)
    lookup: dict[int, tuple[int, int]] = {}
    for rank, index in enumerate(order, start=1):
        lookup[int(index)] = (rank, int(points[index]))
    return lookup


def _archive_row(
    tour: SyntheticTour,
    year: int,
    tourney_id: str,
    name: str,
    surface: str,
    draw_size: int,
    level: str,
    start: pd.Timestamp,
    match_num: int,
    round_name: str,
    best_of: int,
    minutes: int,
    score: str,
    winner: int,
    loser: int,
    w_stats,
    l_stats,
    seeds: dict[int, int],
    rank_lookup: dict[int, tuple[int, int]],
) -> dict:
    """Emit a row using the exact column names of the public archives."""
    row = {
        "tourney_id": tourney_id,
        "tourney_name": name,
        "surface": surface,
        "draw_size": draw_size,
        "tourney_level": level,
        "tourney_date": int(start.strftime("%Y%m%d")),
        "match_num": match_num,
        "score": score,
        "best_of": best_of,
        "round": round_name,
        "minutes": minutes,
    }
    for prefix, player, stats in (("winner", winner, w_stats), ("loser", loser, l_stats)):
        rank, points = rank_lookup.get(player, (999, 0))
        row[f"{prefix}_id"] = int(tour.player_id[player])
        row[f"{prefix}_name"] = tour.name[player]
        row[f"{prefix}_hand"] = tour.hand[player]
        row[f"{prefix}_ht"] = int(tour.height[player])
        row[f"{prefix}_ioc"] = tour.ioc[player]
        row[f"{prefix}_age"] = round(float(tour._age(player, year)), 1)
        row[f"{prefix}_rank"] = rank
        row[f"{prefix}_rank_points"] = points
        row[f"{prefix}_seed"] = seeds.get(player)
        row[f"{prefix}_entry"] = None
        tag = "w" if prefix == "winner" else "l"
        row[f"{tag}_ace"] = stats.ace
        row[f"{tag}_df"] = stats.df
        row[f"{tag}_svpt"] = stats.svpt
        row[f"{tag}_1stIn"] = stats.first_in
        row[f"{tag}_1stWon"] = stats.first_won
        row[f"{tag}_2ndWon"] = stats.second_won
        row[f"{tag}_SvGms"] = stats.sv_gms
        row[f"{tag}_bpSaved"] = stats.bp_saved
        row[f"{tag}_bpFaced"] = stats.bp_faced
    return row


def generate_synthetic_archive(
    tours: tuple[str, ...] = ("atp", "wta"),
    start_year: int = 2005,
    end_year: int = 2024,
    n_players: int = 300,
    seed: int = 20240101,
) -> dict[str, int]:
    """Write synthetic season CSVs into the raw cache, one file per year."""
    counts: dict[str, int] = {}
    for tour in tours:
        world = SyntheticTour(tour, start_year, end_year, n_players=n_players, seed=seed)
        frame = world.generate()
        frame["_year"] = frame["tourney_date"].astype(str).str[:4].astype(int)
        directory = RAW_DIR / tour
        directory.mkdir(parents=True, exist_ok=True)
        for year, chunk in frame.groupby("_year"):
            chunk.drop(columns=["_year"]).to_csv(
                directory / f"{tour}_matches_{year}.csv", index=False
            )
        counts[tour] = len(frame)
        log.info("synthetic %s: %d matches written", tour, len(frame))
    return counts
