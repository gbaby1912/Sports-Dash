"""Player-name normalisation and cross-source resolution.

Every tennis data source names players differently, and joining them is the
single most error-prone part of assembling a real tour dataset:

    Sackmann / Tennis Abstract   "Carlos Alcaraz"          "Felix Auger-Aliassime"
    tennis-data.co.uk (odds)     "Alcaraz C."              "Auger-Aliassime F."
    Tennis Explorer              "Alcaraz Carlos"          "Auger-Aliassime Felix"

Getting this wrong is worse than not joining at all: a mis-linked row silently
attaches one player's odds to another player's match and nothing downstream
complains. The resolver is therefore deliberately conservative - it would rather
leave a row unlinked, and say so, than guess.

Two ideas do the work.

**Name convention is a property of the source, not something to infer.**
"Alcaraz Carlos" and "Carlos Alcaraz" are indistinguishable in isolation, so
each adapter declares which order it uses. Guessing per-name is how you end up
matching "Carlos Alcaraz" to a hypothetical "Alcaraz Carlos" and never noticing.

**One name yields several candidate keys.** Compound surnames are genuinely
ambiguous - is "Beatriz Haddad Maia" surname "Maia" or "Haddad Maia"? Is
"Jan-Lennard Struff" surname "Struff" or "Lennard Struff"? Rather than pick,
each name expands to the set of readings that are plausible under its own
convention, and two names match when their sets intersect. Both sides expand, so
"Struff" (from "Struff J.L.") meets "Struff" (from "Jan-Lennard Struff") even
though the naive parse of each disagrees.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

# Particles that belong to a surname rather than being a given name. Several
# are two letters long, which is exactly why they cannot be told apart from an
# abbreviated given name ("Ka.", "Kr.") by length alone.
SURNAME_PARTICLES = {
    "de", "del", "della", "der", "den", "di", "da", "dos", "das", "du",
    "van", "von", "vander", "ter", "te", "la", "le", "el", "al", "bin",
    "ibn", "mc", "mac", "st", "san", "santa", "do", "af", "auf", "ten",
    "op", "in",
}
# Note: Spanish "y"/"e" ("Fernandez y Garcia") are deliberately NOT listed. A
# lone letter is an abbreviated given name far more often than a particle, and
# treating "Y." as a particle silently breaks every player whose given name
# starts with it.

_SUFFIXES = {"jr", "ii", "iii", "iv", "sr"}
_PUNCT = re.compile(r"[^a-z0-9\s'-]")
_SPACES = re.compile(r"\s+")


class NameOrder(str, Enum):
    """Which end of the string the surname sits on, per source."""

    GIVEN_FIRST = "given_first"      # "Carlos Alcaraz"       (Sackmann)
    SURNAME_FIRST = "surname_first"  # "Alcaraz C." / "Alcaraz Carlos"


# Names no rule handles. Deliberately short: an alias table that grows without
# bound is a sign the rules are wrong, not that the world is irregular.
ALIASES: dict[str, str] = {
    "pedro martinez portero": "pedro martinez",
    "martinez cerezo p.": "martinez p.",
    "kuznetsov an.": "kuznetsov a.",
    "cerundolo j.m.": "cerundolo j.",
    "ramos a.": "ramos vinolas a.",
}


def strip_accents(text: str) -> str:
    """Fold accented characters to ASCII: 'Safarova' from 'Šafářová'."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalise(name: str | None) -> str:
    """Lowercase, de-accent, split on punctuation, collapse whitespace.

    Periods and hyphens become separators rather than being deleted, so
    "J.L." becomes two tokens and "Auger-Aliassime" becomes two tokens. Both
    are then reassembled by the key builder, which is what lets a hyphenated
    given name and a hyphenated surname be treated differently.
    """
    if not name:
        return ""
    text = strip_accents(name).lower().replace("'", "")
    text = text.replace(".", " ").replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _tokens(name: str | None) -> list[str]:
    text = normalise(name)
    if text in ALIASES:
        text = normalise(ALIASES[text])
    return [t for t in text.split() if t and t not in _SUFFIXES]


def is_initial(token: str) -> bool:
    """True for an abbreviated given name ('c', 'ka'), false for a particle.

    Length alone is not enough: 'de' and 'ka' are both two letters, but one is
    part of a surname and the other is an abbreviated given name. A *single*
    letter, though, is always an initial - no surname particle in tennis is one
    character long.
    """
    if not token.isalpha():
        return False
    if len(token) == 1:
        return True
    return len(token) == 2 and token not in SURNAME_PARTICLES


@dataclass(frozen=True)
class NameKey:
    """A (surname, first-initial) pair - all the sparsest format carries."""

    surname: str
    initial: str

    def __str__(self) -> str:
        return f"{self.surname}|{self.initial}"


def _join(tokens: list[str]) -> str:
    return "".join(tokens)


def _surname_readings(surname_tokens: list[str]) -> set[str]:
    """Plausible surnames for a run of tokens, longest and shortest.

    "haddad maia" reads as both "haddadmaia" and "maia"; "van de zandschulp"
    reads as "vandezandschulp" and, because some sources drop the particles,
    "zandschulp" too.
    """
    if not surname_tokens:
        return set()
    readings = {_join(surname_tokens), _join(surname_tokens[-1:])}
    if len(surname_tokens) > 2:
        readings.add(_join(surname_tokens[-2:]))
    # Particle-stripped form, for sources that drop "van"/"de".
    without_particles = [t for t in surname_tokens if t not in SURNAME_PARTICLES]
    if without_particles and without_particles != surname_tokens:
        readings.add(_join(without_particles))
    return {r for r in readings if r}


def candidate_keys(name: str | None, order: NameOrder) -> set[NameKey]:
    """Every (surname, initial) reading of ``name`` under its source's order."""
    tokens = _tokens(name)
    if not tokens:
        return set()
    if len(tokens) == 1:
        return {NameKey(tokens[0], "")}

    initial_positions = [i for i, t in enumerate(tokens) if is_initial(t)]

    if order is NameOrder.SURNAME_FIRST:
        if initial_positions:
            # "Alcaraz C.", "Struff J L", "Van de Zandschulp B."
            # The *first* initial marks where the surname ends. Taking the last
            # one instead swallows the leading initial of a two-initial given
            # name ("Struff J.L." -> surname "struffj", initial "l").
            position = min(
                (i for i in initial_positions if i > 0),
                default=initial_positions[0],
            )
            if position == 0:
                surname_tokens, initial = tokens[1:], tokens[0]
            else:
                surname_tokens, initial = tokens[:position], tokens[position]
        else:
            # "Alcaraz Carlos" - the trailing token is the given name.
            surname_tokens, initial = tokens[:-1], tokens[-1]
    else:
        # "C. Alcaraz", "Carlos Alcaraz", "Jan Lennard Struff", "Beatriz Haddad Maia"
        surname_tokens, initial = tokens[1:], tokens[0]

    if not surname_tokens:
        return {NameKey(_join(tokens), initial[:1])}

    # Emit both a one-letter and (where available) a two-letter initial. Sources
    # that write "Pliskova Ka." and "Pliskova Kr." are handing over a second
    # character of information, and throwing it away is what makes the Pliskova
    # sisters - and every other same-surname, same-initial pair - unresolvable.
    keys: set[NameKey] = set()
    for reading in _surname_readings(surname_tokens):
        keys.add(NameKey(reading, initial[:1]))
        if len(initial) >= 2:
            keys.add(NameKey(reading, initial[:2]))
    return keys


def primary_key(name: str | None, order: NameOrder) -> NameKey | None:
    """The single most likely reading - the longest surname form."""
    keys = candidate_keys(name, order)
    if not keys:
        return None
    return max(keys, key=lambda k: len(k.surname))


@dataclass
class PlayerResolver:
    """Resolves a source's player name to a canonical player id.

    Built from the reference directory (the stats source, which carries full
    names and stable ids) and then queried with names from the odds sources.
    """

    index: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    info: dict[int, tuple[str, float | None]] = field(default_factory=dict)
    unresolved: dict[str, int] = field(default_factory=dict)
    ambiguous: dict[str, int] = field(default_factory=dict)
    resolved_count: int = 0

    @classmethod
    def from_directory(
        cls, directory, order: NameOrder = NameOrder.GIVEN_FIRST
    ) -> "PlayerResolver":
        resolver = cls()
        for row in directory.itertuples(index=False):
            player_id = int(row.player_id)
            rank = getattr(row, "last_rank", None)
            resolver.info[player_id] = (row.name, rank)
            for key in candidate_keys(row.name, order):
                resolver.index.setdefault((row.tour, str(key)), set()).add(player_id)
        return resolver

    def resolve(
        self,
        tour: str,
        name: str,
        order: NameOrder = NameOrder.SURNAME_FIRST,
        rank: float | None = None,
    ) -> int | None:
        """Player id for a source name, or None when not confident.

        ``rank`` is the ranking the source published alongside the name in this
        match. It is the tie-breaker that makes same-surname pairs resolvable,
        and every odds feed publishes it, so it costs nothing to use.
        """
        keys = candidate_keys(name, order)
        if not keys:
            return None

        # Two-letter keys carry strictly more information, so try them first: if
        # they pin down exactly one player, the one-letter keys cannot improve
        # on that and would only reintroduce the ambiguity.
        specific: set[int] = set()
        for key in (k for k in keys if len(k.initial) >= 2):
            specific |= self.index.get((tour, str(key)), set())
        if len(specific) == 1:
            self.resolved_count += 1
            return next(iter(specific))

        candidates: set[int] = set()
        for key in keys:
            candidates |= self.index.get((tour, str(key)), set())

        if not candidates:
            self.unresolved[f"{tour}:{name}"] = self.unresolved.get(f"{tour}:{name}", 0) + 1
            return None
        if len(candidates) == 1:
            self.resolved_count += 1
            return next(iter(candidates))

        chosen = self._break_tie(candidates, rank)
        if chosen is not None:
            self.resolved_count += 1
            return chosen
        self.ambiguous[f"{tour}:{name}"] = self.ambiguous.get(f"{tour}:{name}", 0) + 1
        return None

    def _break_tie(self, candidates: set[int], rank: float | None) -> int | None:
        """Pick between same-key players using the published ranking.

        Refuses unless one candidate is a clear winner: a near-tie means the
        ranking is not actually discriminating and a guess would be a coin flip
        that silently corrupts the join.
        """
        if rank is None or rank != rank:
            return None
        scored = []
        for player_id in candidates:
            known = self.info.get(player_id, (None, None))[1]
            if known is None or known != known:
                continue
            scored.append((abs(float(known) - float(rank)), player_id))
        if not scored:
            return None
        scored.sort()
        if len(scored) == 1:
            return scored[0][1]
        # Require a clear separation before trusting the tie-break.
        if scored[0][0] <= 10 and scored[1][0] > scored[0][0] * 3:
            return scored[0][1]
        return None

    def report(self) -> dict:
        """Diagnostics, so a bad join is visible instead of silent."""
        attempted = self.resolved_count + sum(self.unresolved.values()) + sum(
            self.ambiguous.values()
        )
        return {
            "indexed_keys": len(self.index),
            "players": len(self.info),
            "resolved_rows": self.resolved_count,
            "attempted_rows": attempted,
            "resolve_rate": self.resolved_count / attempted if attempted else 0.0,
            "unresolved_names": len(self.unresolved),
            "ambiguous_names": len(self.ambiguous),
            "worst_unresolved": sorted(self.unresolved.items(), key=lambda kv: -kv[1])[:20],
            "worst_ambiguous": sorted(self.ambiguous.items(), key=lambda kv: -kv[1])[:20],
        }
