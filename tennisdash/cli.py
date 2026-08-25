"""Command-line interface for the whole pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data import sources

    failures = []
    for tour in args.tours:
        report = sources.fetch_matches(tour, start_year=args.start_year, end_year=args.end_year)
        print(f"{tour}: {report.summary()}")
        for name, reason in report.failed[:5]:
            print(f"    ! {name}: {reason}")
        failures.extend(report.failed)
        sources.fetch_players(tour)
        sources.fetch_rankings(tour)
        if args.odds:
            paths = sources.fetch_odds(tour, start_year=max(args.start_year, 2010))
            print(f"{tour}: {len(paths)} odds files")

    if failures and not args.allow_partial:
        print(
            "\nSome archives could not be downloaded. If these are HTTP 403/404 the host is\n"
            "blocked by your network policy - run this step from a machine with access, or\n"
            "use `tennisdash synth` to generate an offline dataset instead.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which data sources this machine can actually reach."""
    from .data.sources.registry import diagnose, format_report

    results = diagnose()
    print(format_report(results))
    return 0 if all(e["reachable"] for e in results if e["required"]) else 2


def cmd_synth(args: argparse.Namespace) -> int:
    from .data.synthetic import generate_synthetic_archive

    counts = generate_synthetic_archive(
        tours=tuple(args.tours),
        start_year=args.start_year,
        end_year=args.end_year,
        n_players=args.players,
        seed=args.seed,
    )
    for tour, count in counts.items():
        print(f"{tour}: {count} synthetic matches written to the raw cache")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from .data.ingest import build_match_table

    frame = build_match_table(tours=tuple(args.tours))
    if frame.empty:
        print("No raw data found. Run `tennisdash fetch` or `tennisdash synth` first.",
              file=sys.stderr)
        return 1
    print(f"{len(frame)} matches  {frame.match_date.min().date()} -> {frame.match_date.max().date()}")
    print(frame.groupby("tour").size().to_string())
    print(f"serve-stat coverage: {frame.has_serve_stats.mean():.1%}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    import joblib

    from .config import ARTIFACT_DIR
    from .data.ingest import load_match_table
    from .features.builder import build_features, feature_columns

    features, engines = build_features(load_match_table())
    joblib.dump(engines, ARTIFACT_DIR / "engines.joblib")
    print(f"{len(features)} rows x {len(feature_columns(features))} model features")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    bundle = train(
        rebuild_features=args.rebuild,
        backtest=not args.no_backtest,
        importance=not args.no_importance,
    )
    meta = bundle["metadata"]
    print(f"\nTrained on {meta['training_rows']} matches through {meta['trained_through']}")
    print(f"Features: {meta['n_features']}   Calibration: {meta['calibration_method']}")
    pooled = meta["report"].get("backtest_pooled", [])
    overall = next((r for r in pooled if r["group"] == "all"), None)
    if overall:
        print(
            f"Walk-forward (out-of-sample, n={overall['n']}): "
            f"log loss {overall['log_loss']:.4f}  accuracy {overall['accuracy']:.4f}  "
            f"ECE {overall['ece']:.4f}  vs Elo baseline {overall['ref_elo_log_loss']:.4f}"
        )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest.walkforward import pooled_summary, walk_forward
    from .train import load_or_build_features

    features, _ = load_or_build_features()
    summary, _ = walk_forward(features, start_year=args.start_year)
    print("\n--- by season (all tours) ---")
    seasonal = summary[summary.tour == "all"]
    print(seasonal[["year", "n", "log_loss", "elo_log_loss", "accuracy", "ece",
                    "calibration_slope", "skill_vs_elo"]].round(4).to_string(index=False))
    print("\n--- pooled out-of-sample ---")
    pooled = pooled_summary(walk_forward.predictions)
    print(pooled[["group", "n", "log_loss", "brier", "accuracy", "ece",
                  "calibration_slope", "ref_elo_log_loss", "skill_vs_elo"]].round(4).to_string(index=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import MatchContext, MatchPredictor
    from .train import load_bundle

    predictor = MatchPredictor(load_bundle())

    def resolve(query: str) -> int:
        if query.isdigit():
            return int(query)
        found = predictor.find(args.tour, query)
        if found.empty:
            raise SystemExit(f"no player matching {query!r} on the {args.tour.upper()} tour")
        return int(found.iloc[0].player_id)

    p1, p2 = resolve(args.player1), resolve(args.player2)
    context = MatchContext(
        surface=args.surface,
        best_of=args.best_of,
        level=args.level,
        round=args.round,
        tourney_name=args.tournament,
        indoor=args.indoor,
    )
    prediction = predictor.predict(args.tour, p1, p2, context)

    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2, default=str))
        return 0

    print(f"\n{prediction.p1_name}  vs  {prediction.p2_name}")
    print(f"{args.surface}, best of {args.best_of}, {args.tournament}\n")
    print(f"  {prediction.p1_name:<28s} {prediction.probability:6.1%}")
    print(f"  {prediction.p2_name:<28s} {1 - prediction.probability:6.1%}\n")
    if prediction.serve:
        s = prediction.serve
        print(f"  Expected serve points won:  {s['p1_expected_spw']:.1%} / {s['p2_expected_spw']:.1%}")
        print(f"  Projected hold rate:        {s['p1_hold_pct']:.1%} / {s['p2_hold_pct']:.1%}")
    if prediction.scores:
        print("\n  Most likely scorelines:")
        for score, probability in sorted(prediction.scores.items(), key=lambda kv: -kv[1])[:4]:
            print(f"    {score}   {probability:5.1%}")
    if prediction.factors:
        print("\n  Biggest factors:")
        for factor in prediction.factors[:6]:
            who = prediction.p1_name if factor["favours"] == "p1" else prediction.p2_name
            print(f"    {factor['label']:<38s} {abs(factor['contribution']):5.2f} -> {who}")
    print()
    return 0


def cmd_rankings(args: argparse.Namespace) -> int:
    from .train import load_bundle

    bundle = load_bundle()
    engine = bundle["engines"]["elo"]
    directory = bundle["directory"]
    snapshot = engine.snapshot(args.tour, surface=args.surface)
    snapshot = snapshot[snapshot["matches"] >= args.min_matches]
    names = directory[directory.tour == args.tour].set_index("player_id")["name"]
    snapshot["name"] = snapshot["player_id"].map(names)
    column = f"elo_{args.surface.lower()}" if args.surface else "elo"
    print(snapshot.head(args.top)[["name", column, "elo", "matches", "peak_elo"]]
          .round(1).to_string(index=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("tennisdash.api.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tennisdash",
        description="ATP/WTA match prediction engine and dashboard",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    tours = ["atp", "wta"]

    fetch = sub.add_parser("fetch", help="download the public match archives")
    fetch.add_argument("--tours", nargs="+", default=tours)
    fetch.add_argument("--start-year", type=int, default=2000)
    fetch.add_argument("--end-year", type=int, default=None)
    fetch.add_argument("--odds", action="store_true", help="also fetch closing odds for benchmarking")
    fetch.add_argument("--allow-partial", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    doctor = sub.add_parser("doctor", help="check which data sources are reachable")
    doctor.set_defaults(func=cmd_doctor)

    synth = sub.add_parser(
        "synth",
        help="generate a synthetic dataset (TEST FIXTURE ONLY - not tour data)",
    )
    synth.add_argument("--tours", nargs="+", default=tours)
    synth.add_argument("--start-year", type=int, default=2005)
    synth.add_argument("--end-year", type=int, default=2024)
    synth.add_argument("--players", type=int, default=320)
    synth.add_argument("--seed", type=int, default=20240101)
    synth.set_defaults(func=cmd_synth)

    ingest = sub.add_parser("ingest", help="normalise raw archives into the match table")
    ingest.add_argument("--tours", nargs="+", default=tours)
    ingest.set_defaults(func=cmd_ingest)

    features = sub.add_parser("features", help="build the feature matrix")
    features.set_defaults(func=cmd_features)

    train = sub.add_parser("train", help="train the model and write the bundle")
    train.add_argument("--rebuild", action="store_true", help="rebuild features from scratch")
    train.add_argument("--no-backtest", action="store_true")
    train.add_argument("--no-importance", action="store_true")
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest", help="walk-forward evaluation")
    backtest.add_argument("--start-year", type=int, default=None)
    backtest.set_defaults(func=cmd_backtest)

    predict = sub.add_parser("predict", help="predict one matchup")
    predict.add_argument("player1")
    predict.add_argument("player2")
    predict.add_argument("--tour", default="atp", choices=tours)
    predict.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    predict.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    predict.add_argument("--level", default="A")
    predict.add_argument("--round", default="R32")
    predict.add_argument("--tournament", default="Neutral Court")
    predict.add_argument("--indoor", action="store_true", default=None)
    predict.add_argument("--json", action="store_true")
    predict.set_defaults(func=cmd_predict)

    rankings = sub.add_parser("rankings", help="Elo leaderboard")
    rankings.add_argument("--tour", default="atp", choices=tours)
    rankings.add_argument("--surface", default=None, choices=["Hard", "Clay", "Grass", "Carpet"])
    rankings.add_argument("--top", type=int, default=25)
    rankings.add_argument("--min-matches", type=int, default=20)
    rankings.set_defaults(func=cmd_rankings)

    serve = sub.add_parser("serve", help="run the dashboard API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    pd.set_option("display.width", 200)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
