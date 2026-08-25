"""Data source adapters.

Each module knows how to fetch and normalise one source. They are kept separate
because they fail independently: a blocked host, a changed HTML layout or a
season that has not been published yet should degrade one source, not the
pipeline.

    sackmann        results + per-match serve statistics (the modelling backbone)
    tennis_data     closing odds, indoor/outdoor flag, series (the market view)
    tennis_explorer results + odds, as a cross-check and a gap-filler
"""
from . import http, sackmann, tennis_data  # noqa: F401

__all__ = ["http", "sackmann", "tennis_data"]
