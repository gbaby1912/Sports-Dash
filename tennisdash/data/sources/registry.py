"""Which sources exist, what each provides, and whether it is reachable.

The `tennisdash doctor` command reports this. It exists because "the pipeline
produced no data" is a useless error message: the cause is almost always one
specific blocked host, and the fix is different for each one.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import sackmann, tennis_data
from .http import check_host


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    host: str
    probe_url: str
    provides: str
    required: bool
    remedy: str


SOURCES = [
    SourceSpec(
        key="sackmann",
        name="Jeff Sackmann archives (Tennis Abstract)",
        host="raw.githubusercontent.com",
        probe_url=sackmann.matches_url("atp", 2024),
        provides="results + per-match serve stats (aces, service points, break points)",
        required=True,
        remedy=(
            "This is the modelling backbone and the pipeline cannot train without it.\n"
            "     Options, easiest first:\n"
            "       1. Fork github.com/JeffSackmann/tennis_atp and tennis_wta into your\n"
            "          own GitHub account, then they can be attached to this session\n"
            "          (same-owner repositories are allowed; other owners are not).\n"
            "       2. Add raw.githubusercontent.com to this environment's egress\n"
            "          allowlist - see code.claude.com/docs/en/claude-code-on-the-web\n"
            "       3. Clone the repos and run `make data` on your own machine."
        ),
    ),
    SourceSpec(
        key="tennis_data",
        name="tennis-data.co.uk",
        host=tennis_data.HOST,
        probe_url=tennis_data.season_url("atp", 2024),
        provides="closing odds (Pinnacle, Bet365, max, average), indoor flag, series",
        required=False,
        remedy=(
            "Without it the model still trains, but it cannot be benchmarked against\n"
            "     the betting market and no edge/ROI figures are available.\n"
            "     Add www.tennis-data.co.uk to the egress allowlist, or fetch locally."
        ),
    ),
]


def diagnose(timeout: int = 15) -> list[dict]:
    """Probe every source and describe what is and is not available."""
    results = []
    for spec in SOURCES:
        reachable, detail = check_host(spec.probe_url, timeout=timeout)
        results.append(
            {
                "key": spec.key,
                "name": spec.name,
                "host": spec.host,
                "provides": spec.provides,
                "required": spec.required,
                "reachable": reachable,
                "detail": detail,
                "remedy": None if reachable else spec.remedy,
            }
        )
    return results


def format_report(results: list[dict]) -> str:
    lines = ["", "Data source availability", "=" * 66]
    for entry in results:
        mark = "OK    " if entry["reachable"] else "BLOCKED"
        tag = "required" if entry["required"] else "optional"
        lines.append(f"  [{mark}] {entry['name']}  ({tag})")
        lines.append(f"            host:     {entry['host']}")
        lines.append(f"            provides: {entry['provides']}")
        lines.append(f"            status:   {entry['detail']}")
        if entry["remedy"]:
            lines.append(f"            fix:      {entry['remedy']}")
        lines.append("")

    blocked_required = [e for e in results if e["required"] and not e["reachable"]]
    if blocked_required:
        lines.append("A required source is unreachable, so `make data` cannot build a real")
        lines.append("dataset from this machine. The synthetic generator exists only to keep")
        lines.append("the test suite runnable; it is not a substitute for tour data.")
    elif all(e["reachable"] for e in results):
        lines.append("All sources reachable. Run `make data && make train`.")
    else:
        lines.append("Required sources reachable. Optional ones are missing, so the model")
        lines.append("will train but cannot be benchmarked against the market.")
    return "\n".join(lines)
