"""Venue metadata that the public match archives do not carry.

Indoor/outdoor and altitude are two of the strongest *context* effects in
tennis and neither is present in the Sackmann archives, so they are curated
here. Both matter for real reasons:

* **Indoor** removes wind and sun and keeps the ball flying consistently, which
  systematically favours flat, big-serving, first-strike players and penalises
  heavy-topspin grinders. The same two players can have materially different
  win probabilities indoors vs outdoors on nominally identical hard courts.
* **Altitude** thins the air, so the ball travels faster and bounces higher.
  Serve dominance rises sharply: hold percentages at Bogota (2 640 m) or Quito
  (2 850 m) run several points above sea-level norms on the same surface.

Matching is done on a normalised substring of the tournament name, which is
stable across the archives even as sponsor names change.
"""
from __future__ import annotations

import re

# Tournament-name fragment -> approximate venue elevation in metres.
ALTITUDE_M: dict[str, int] = {
    "bogota": 2640,
    "quito": 2850,
    "mexico city": 2240,
    "puebla": 2135,
    "guadalajara": 1566,
    "la paz": 3640,
    "cali": 1018,
    "bucaramanga": 959,
    "medellin": 1495,
    "gstaad": 1050,
    "kitzbuhel": 762,
    "kitzbuehel": 762,
    "madrid": 667,
    "johannesburg": 1753,
    "pretoria": 1339,
    "denver": 1609,
    "aspen": 2438,
    "vail": 2445,
    "nairobi": 1795,
    "addis": 2355,
    "almaty": 786,
    "santiago": 570,
    "sao paulo": 760,
    "campinas": 686,
    "curitiba": 935,
    "belo horizonte": 852,
    "brasilia": 1172,
    "asuncion": 43,
    "kunming": 1892,
    "anning": 1800,
    "shymkent": 506,
    "tashkent": 455,
    "isfahan": 1590,
    "tehran": 1200,
    "srinagar": 1585,
    "calgary": 1045,
    "salt lake": 1288,
    "albuquerque": 1619,
}

# Tournament-name fragments that are (or historically were) played indoors.
INDOOR_TOURNAMENTS: set[str] = {
    # --- ATP ---
    "paris masters", "bercy", "rolex paris",
    "rotterdam", "marseille", "montpellier", "metz", "moselle",
    "vienna", "basel", "stockholm", "antwerp", "sofia", "zagreb",
    "st petersburg", "st. petersburg", "moscow", "kremlin",
    "atp finals", "masters cup", "tour finals", "world tour finals",
    "next gen", "nextgen",
    "milan", "memphis", "san jose", "copenhagen", "rennes", "cologne",
    "bratislava", "lyon indoor", "dallas", "gijon", "naples", "florence",
    "tel aviv", "nur-sultan", "astana", "singapore indoor",
    "bercy-paris", "eindhoven", "arnhem", "bolzano", "bergamo",
    # --- WTA ---
    "linz", "luxembourg", "quebec", "zurich", "ostrava",
    "cluj", "courmayeur", "diamond games", "gdf suez",
    "open gaz de france", "wta finals", "wta elite trophy",
    "hua hin indoor", "budapest indoor", "portoroz indoor",
    # Stuttgart's WTA event is indoor clay - a genuinely distinctive surface.
    "stuttgart",
}

# Explicit exclusions: names that would otherwise collide with an indoor entry.
_INDOOR_EXCLUSIONS = {
    "paris",          # Roland Garros is listed as "Roland Garros", but guard anyway
    "roland garros",
    "lyon",           # modern Lyon is outdoor clay; "lyon indoor" is matched above
    "milan indoor clay",
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalise_name(name: str | None) -> str:
    if not name:
        return ""
    text = _NON_ALNUM.sub(" ", str(name).lower())
    return re.sub(r"\s+", " ", text).strip()


def venue_altitude(tourney_name: str | None) -> int:
    """Approximate elevation in metres for a tournament, 0 if unknown."""
    key = normalise_name(tourney_name)
    if not key:
        return 0
    for fragment, metres in ALTITUDE_M.items():
        if fragment in key:
            return metres
    return 0


def is_indoor(tourney_name: str | None, surface: str | None = None) -> bool:
    """Best-effort indoor flag for a tournament.

    Carpet is always indoors. Otherwise we match against the curated list.
    """
    if surface and str(surface).strip().lower() == "carpet":
        return True
    key = normalise_name(tourney_name)
    if not key:
        return False
    if key in _INDOOR_EXCLUSIONS:
        return False
    if "indoor" in key:
        return True
    return any(fragment in key for fragment in INDOOR_TOURNAMENTS)


# Tournament-name fragment -> host country IOC code, for a home-crowd feature.
# Home advantage in tennis is real but modest (a couple of percentage points),
# and it is strongest at smaller events where the crowd is one-sided.
TOURNAMENT_COUNTRY: dict[str, str] = {
    "australian open": "AUS", "brisbane": "AUS", "sydney": "AUS", "adelaide": "AUS",
    "melbourne": "AUS", "perth": "AUS",
    "roland garros": "FRA", "french open": "FRA", "paris": "FRA", "marseille": "FRA",
    "montpellier": "FRA", "metz": "FRA", "lyon": "FRA", "rennes": "FRA", "strasbourg": "FRA",
    "wimbledon": "GBR", "queens": "GBR", "eastbourne": "GBR", "birmingham": "GBR",
    "nottingham": "GBR", "london": "GBR",
    "us open": "USA", "indian wells": "USA", "miami": "USA", "cincinnati": "USA",
    "washington": "USA", "atlanta": "USA", "winston": "USA", "newport": "USA",
    "delray": "USA", "san jose": "USA", "memphis": "USA", "houston": "USA", "dallas": "USA",
    "charleston": "USA", "stanford": "USA", "san diego": "USA", "austin": "USA",
    "madrid": "ESP", "barcelona": "ESP", "valencia": "ESP", "mallorca": "ESP",
    "marbella": "ESP", "estoril": "POR",
    "rome": "ITA", "florence": "ITA", "naples": "ITA", "turin": "ITA", "milan": "ITA",
    "palermo": "ITA", "sardinia": "ITA", "parma": "ITA",
    "hamburg": "GER", "stuttgart": "GER", "munich": "GER", "halle": "GER", "berlin": "GER",
    "cologne": "GER", "bad homburg": "GER",
    "monte carlo": "MON", "basel": "SUI", "geneva": "SUI", "gstaad": "SUI",
    "vienna": "AUT", "kitzbuhel": "AUT", "kitzbuehel": "AUT", "linz": "AUT",
    "rotterdam": "NED", "s-hertogenbosch": "NED", "amersfoort": "NED", "eindhoven": "NED",
    "antwerp": "BEL", "brussels": "BEL",
    "stockholm": "SWE", "bastad": "SWE", "copenhagen": "DEN",
    "moscow": "RUS", "st petersburg": "RUS", "kremlin": "RUS",
    "beijing": "CHN", "shanghai": "CHN", "zhuhai": "CHN", "wuhan": "CHN", "guangzhou": "CHN",
    "tokyo": "JPN", "osaka": "JPN",
    "toronto": "CAN", "montreal": "CAN", "canada": "CAN",
    "rio de janeiro": "BRA", "sao paulo": "BRA", "brasilia": "BRA",
    "buenos aires": "ARG", "cordoba": "ARG",
    "santiago": "CHI", "bogota": "COL", "quito": "ECU", "lima": "PER",
    "acapulco": "MEX", "los cabos": "MEX", "mexico city": "MEX", "guadalajara": "MEX",
    "umag": "CRO", "zagreb": "CRO", "belgrade": "SRB", "budapest": "HUN",
    "prague": "CZE", "ostrava": "CZE", "bratislava": "SVK", "cluj": "ROU", "bucharest": "ROU",
    "sofia": "BUL", "warsaw": "POL", "gdynia": "POL", "istanbul": "TUR", "antalya": "TUR",
    "dubai": "UAE", "doha": "QAT", "tel aviv": "ISR", "chennai": "IND", "pune": "IND",
    "auckland": "NZL", "seoul": "KOR", "hong kong": "HKG", "astana": "KAZ",
    "nur-sultan": "KAZ", "almaty": "KAZ", "tashkent": "UZB", "athens": "GRE",
    "luxembourg": "LUX", "quebec": "CAN", "monterrey": "MEX", "bogota open": "COL",
}


def venue_country(tourney_name: str | None) -> str | None:
    """Host country (IOC code) for a tournament, or None if unknown."""
    key = normalise_name(tourney_name)
    if not key:
        return None
    # Prefer the longest matching fragment so "mexico city" beats "mexico".
    best: tuple[int, str] | None = None
    for fragment, code in TOURNAMENT_COUNTRY.items():
        if fragment in key and (best is None or len(fragment) > best[0]):
            best = (len(fragment), code)
    return best[1] if best else None
