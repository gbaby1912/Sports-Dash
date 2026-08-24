"""Parser for the free-text score strings used by the public match archives.

Scorelines carry real predictive information that a bare win/loss label throws
away: games won, sets dropped, tiebreaks, and whether the match went to a
deciding set. They are also messy - retirements, walkovers, defaults, extended
tiebreaks and match tiebreaks all appear - so parsing is done defensively and
every field is optional.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_SET_RE = re.compile(r"^(\d{1,2})-(\d{1,2})(?:[\(\[]([^\)\]]*)[\)\]])?$")
# Events that replace the deciding set with a match tiebreak write it as a
# bare bracketed score, e.g. "6-3 6-7(5) [10-8]".
_MATCH_TB_RE = re.compile(r"^[\(\[](\d{1,2})-(\d{1,2})[\)\]]$")
_RET_TOKENS = {"ret", "ret.", "retired", "rtd", "rtd."}
_WO_TOKENS = {"w/o", "wo", "walkover", "def", "def.", "default", "defaulted"}
_UNFINISHED_TOKENS = {"abn", "abn.", "abandoned", "unfinished", "in", "progress"}


@dataclass
class ParsedScore:
    """Structured view of a scoreline, from the winner's perspective."""

    winner_games: int = 0
    loser_games: int = 0
    winner_sets: int = 0
    loser_sets: int = 0
    sets_played: int = 0
    tiebreaks_played: int = 0
    winner_tiebreaks: int = 0
    went_to_decider: bool = False
    retirement: bool = False
    walkover: bool = False
    partial_set: bool = False
    parsed: bool = False

    @property
    def completed(self) -> bool:
        return self.parsed and not self.retirement and not self.walkover

    @property
    def game_share(self) -> float:
        """Winner's share of all games played; 0.5 when unknown."""
        total = self.winner_games + self.loser_games
        if total <= 0:
            return 0.5
        return self.winner_games / total

    @property
    def dominance(self) -> float:
        """A bounded [0, 1] measure of how one-sided the match was.

        Combines set margin and game share so that 6-0 6-0 and 7-6 7-6 are
        clearly separated, while 6-4 6-4 sits in between.
        """
        if not self.parsed or self.sets_played == 0:
            return 0.5
        set_margin = (self.winner_sets - self.loser_sets) / max(self.sets_played, 1)
        return float(min(1.0, max(0.0, 0.5 * self.game_share + 0.5 * (0.5 + set_margin / 2))))


def parse_score(score: str | float | None, best_of: int | None = None) -> ParsedScore:
    """Parse a scoreline into games, sets, tiebreaks and completion status."""
    result = ParsedScore()
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return result

    text = str(score).strip()
    if not text:
        return result

    lowered = text.lower()
    if any(tok in lowered.split() or lowered == tok for tok in _WO_TOKENS):
        result.walkover = True
        result.parsed = True
        return result
    # "W/O" sometimes appears without spaces around it.
    if "w/o" in lowered or lowered.startswith("def"):
        result.walkover = True
        result.parsed = True
        return result

    tokens = text.replace(",", " ").split()
    for token in tokens:
        clean = token.strip().lower()
        if clean in _RET_TOKENS:
            result.retirement = True
            continue
        if clean in _WO_TOKENS:
            result.walkover = True
            continue
        if clean in _UNFINISHED_TOKENS:
            continue

        # Some archives write bracketed tiebreak scores detached, e.g. "7-6 (5)".
        stripped = token.strip()
        match = _SET_RE.match(stripped)
        bare_tiebreak = None if match else _MATCH_TB_RE.match(stripped)
        if not match and not bare_tiebreak:
            continue

        if bare_tiebreak is not None:
            # A match tiebreak played instead of a final set: it decides a set,
            # counts as a tiebreak, but contributes only one game to each side's
            # game tally rather than its raw point score.
            pa, pb = int(bare_tiebreak.group(1)), int(bare_tiebreak.group(2))
            result.sets_played += 1
            result.tiebreaks_played += 1
            if pa > pb:
                result.winner_sets += 1
                result.winner_games += 1
                result.winner_tiebreaks += 1
            else:
                result.loser_sets += 1
                result.loser_games += 1
            continue

        w_games, l_games = int(match.group(1)), int(match.group(2))
        # Guard against corrupt rows such as "60-0".
        if w_games > 30 or l_games > 30:
            continue

        high, low = max(w_games, l_games), min(w_games, l_games)
        # A match tiebreak standing in for a deciding set is written like
        # "10-8"; a regular set needs 6 games and a two-game margin, or a
        # 7-5 / 7-6 finish.
        is_match_tiebreak = high >= 10 and result.sets_played >= 2
        is_complete_set = (
            (high >= 6 and high - low >= 2) or (high == 7 and low in (5, 6)) or is_match_tiebreak
        )

        result.winner_games += w_games
        result.loser_games += l_games
        result.sets_played += 1
        # A partial set abandoned at a retirement contributes its games but is
        # not credited to either player as a set won.
        if is_complete_set:
            if w_games > l_games:
                result.winner_sets += 1
            elif l_games > w_games:
                result.loser_sets += 1
        else:
            result.partial_set = True

        if match.group(3) is not None or is_match_tiebreak or {w_games, l_games} == {7, 6}:
            result.tiebreaks_played += 1
            if w_games > l_games:
                result.winner_tiebreaks += 1

    result.parsed = result.sets_played > 0 or result.retirement or result.walkover

    if best_of and result.sets_played >= best_of:
        result.went_to_decider = result.completed and result.sets_played == best_of
    elif result.completed:
        # Fall back on the set count: 3 sets in a Bo3, 5 in a Bo5.
        result.went_to_decider = result.sets_played in (3, 5) and result.loser_sets == (
            result.sets_played // 2
        )

    return result
