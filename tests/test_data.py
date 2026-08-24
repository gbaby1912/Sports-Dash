"""Ingestion, score parsing and venue metadata."""
import pytest

from tennisdash.data.score import parse_score
from tennisdash.data.venues import is_indoor, venue_altitude, venue_country


class TestScoreParsing:
    def test_straight_sets(self):
        parsed = parse_score("6-4 6-3", best_of=3)
        assert (parsed.winner_sets, parsed.loser_sets) == (2, 0)
        assert (parsed.winner_games, parsed.loser_games) == (12, 7)
        assert parsed.completed and not parsed.went_to_decider

    def test_tiebreaks_are_counted(self):
        parsed = parse_score("7-6(5) 6-7(3) 7-6(4)", best_of=3)
        assert parsed.tiebreaks_played == 3
        assert parsed.winner_tiebreaks == 2
        assert parsed.went_to_decider

    def test_extended_tiebreak_bracket(self):
        """`7-6(10-8)` must parse - a plain \\d{1,3} bracket rejects it."""
        parsed = parse_score("7-5 6-7(4) 7-6(10-8)", best_of=3)
        assert parsed.sets_played == 3
        assert parsed.winner_sets == 2
        assert parsed.winner_games == 20

    def test_square_bracket_match_tiebreak(self):
        parsed = parse_score("6-3 6-7(5) [10-8]", best_of=3)
        assert parsed.sets_played == 3
        assert parsed.winner_sets == 2

    def test_retirement_does_not_credit_a_partial_set(self):
        """A player leading 2-0 in games has not won that set."""
        parsed = parse_score("6-2 2-0 RET", best_of=3)
        assert parsed.retirement
        assert parsed.winner_sets == 1
        assert parsed.partial_set
        assert not parsed.completed

    @pytest.mark.parametrize("text", ["W/O", "Def.", "walkover"])
    def test_walkovers(self, text):
        parsed = parse_score(text, best_of=3)
        assert parsed.walkover and not parsed.completed

    @pytest.mark.parametrize("text", [None, "", float("nan"), "unknown"])
    def test_junk_is_survived(self, text):
        parsed = parse_score(text, best_of=3)
        assert parsed.dominance == 0.5

    def test_dominance_ordering(self):
        blowout = parse_score("6-0 6-0", best_of=3).dominance
        close = parse_score("7-6(5) 6-7(3) 7-6(4)", best_of=3).dominance
        routine = parse_score("6-4 6-4", best_of=3).dominance
        assert blowout > routine > close


class TestVenues:
    def test_indoor_detection(self):
        assert is_indoor("Paris Masters", "Hard")
        assert is_indoor("Rotterdam", "Hard")
        assert is_indoor("Anything", "Carpet"), "carpet is always indoors"
        assert not is_indoor("Roland Garros", "Clay")
        assert not is_indoor("Wimbledon", "Grass")

    def test_altitude(self):
        assert venue_altitude("Bogota") > 2000
        assert venue_altitude("Quito") > 2000
        assert venue_altitude("Madrid Masters") == 667
        assert venue_altitude("Wimbledon") == 0

    def test_country_prefers_the_longest_match(self):
        assert venue_country("Mexico City") == "MEX"
        assert venue_country("Roland Garros") == "FRA"
        assert venue_country("Somewhere Unknown") is None


class TestIngest:
    def test_return_points_are_the_complement_of_the_opponents_serve(self, small_matches):
        usable = small_matches[small_matches.has_serve_stats]
        assert len(usable) > 100
        assert (usable.winner_spw + usable.loser_rpw == usable.winner_svpt).all()
        assert (usable.loser_spw + usable.winner_rpw == usable.loser_svpt).all()

    def test_percentages_are_in_range(self, small_matches):
        usable = small_matches[small_matches.has_serve_stats]
        for column in ("winner_spw_pct", "loser_spw_pct", "winner_rpw_pct", "loser_rpw_pct"):
            values = usable[column].dropna()
            assert values.between(0, 1).all(), f"{column} out of range"

    def test_matches_are_chronological_and_unique(self, small_matches):
        assert small_matches.match_date.is_monotonic_increasing
        assert small_matches.match_id.is_unique

    def test_rounds_are_ordered_within_an_event(self, small_matches):
        """A final must not be dated before a first round of the same event."""
        event = small_matches[small_matches.tourney_id == small_matches.tourney_id.iloc[0]]
        first = event[event["round"] == event["round"].iloc[0]]
        final = event[event["round"] == "F"]
        if not final.empty:
            assert final.match_date.iloc[0] >= first.match_date.iloc[0]
