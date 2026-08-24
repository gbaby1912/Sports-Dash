# Sports-Dash — ATP & WTA match predictor

A calibrated tennis match prediction engine and dashboard. It combines four
independent views of a matchup — an Elo rating family, an opponent-adjusted
serve/return model run through the tennis scoring system, gradient-boosted
trees, and a regularised linear model — into a single stacked, calibrated
probability.

```
make install && make data && make train && make serve   # real archives
make install && make demo && make serve                 # offline, synthetic data
```

Then open <http://127.0.0.1:8000>.

---

## What makes it accurate

Most tennis models stop at "Elo, plus a surface adjustment". The gains here come
from four places, in rough order of how much they matter.

### 1. Four ratings, not one

| Rating | Question it answers | Why it earns its place |
|---|---|---|
| `overall` | Who has been winning? | The strongest single baseline in tennis. |
| `surface` | Who wins *on this surface*? | Clay and grass specialists are real; a player's clay and hard ratings can sit 200+ points apart. |
| `points` | Who wins more *points*? | A match is ~150 points, not one binary trial. This rating has far lower variance and reacts faster to a genuine change in level — it is the single most important feature in the fitted model. |
| `games` | Who wins more *games*? | The same idea one level up, and available for older matches that lack point-level stats. |

All four use a match-count-dependent K factor (`K = 250/(n+5)^0.4`), so a rising
junior's rating catches up within a season while a veteran's stays stable. Ratings
regress toward the mean during long layoffs, because a rating is a claim about
*current* level and players overwhelmingly return below their old one.

The surface rating is not used raw. It is blended toward overall Elo with a weight
that rises with how much the player has actually played on that surface — a player
with 15 career grass matches should not get a confident grass rating.

### 2. Opponent-adjusted serve and return

Raw serve-points-won is one of the most-quoted tennis stats and one of the most
misleading, because it is schedule-contaminated: a player who spent the season
drawing elite returners looks worse than they are.

The fix is to treat each match as an observation of a *difference* and fit

```
logit(spw_ij) = μ[tour, surface] + serve_i + serve_i@surface
                                 − return_j − return_j@surface
```

by ridge-penalised weighted least squares — the same idea as adjusted plus-minus
in basketball. Observations are weighted by service points played and decayed
with a one-year half-life; the ridge penalty is expressed in *service points*, so
shrinkage toward tour average is automatic and continuous rather than a cutoff.

**This measurably beats the raw statistic.** Predicting the serve percentage of
future matches, out of sample:

| Predictor | RMSE |
|---|---|
| Tour average for the surface | 0.0773 |
| Raw career serve % (schedule-contaminated) | 0.0628 |
| **Ridge, opponent-adjusted** | **0.0579** |

### 3. A point-based model that is genuinely independent of Elo

Given each player's probability of winning a point on serve, the probability of
winning a game, a tiebreak, a set and the match all follow in closed form. Elo
asks *"who has been winning?"*; this asks *"given how these two serve and return,
who should win?"*. They make different mistakes, which is exactly why stacking
them helps.

The implementation is validated two ways: against textbook hold probabilities
(matching to four decimal places), and against a completely separate
point-by-point simulator that shares no code with it.

### 4. Everything else that actually moves the needle

Grouped as the dashboard presents them:

- **Form** — exponentially decayed recent results, credited by *who* was beaten
  rather than just how many, so beating the world #3 counts for more than beating
  a qualifier.
- **Rest and workload** — court time and matches in the last 7/14/28 days, days
  since the last match, five-setters survived, recent retirements as an injury
  proxy. The rest relationship is U-shaped, so distance from optimal rest is given
  explicitly alongside the raw number.
- **Clutch** — break points saved and converted, tiebreak record, deciding-set
  record. These are the most over-interpreted numbers in tennis, so each is shrunk
  hard toward tour average with an explicit pseudo-count.
- **Surface context** — career record on the surface, and how much of the *recent*
  schedule was on it (results right after a surface switch are noisier).
- **Head-to-head** — shrunk heavily toward even. Most pairs have played fewer than
  four times, and H2H adds little once the rating gap is controlled for. It is
  present so a genuinely long rivalry can register, not so that a 1–0 record from
  2019 can swing a prediction.
- **Conditions** — indoor/outdoor and altitude, neither of which the public
  archives carry, so both are curated in `tennisdash/data/venues.py`. Indoor
  courts remove wind and sun and favour flat first-strike players; thin air at
  Bogotá (2 640 m) or Quito (2 850 m) pushes hold percentages several points above
  sea-level norms.
- **Player attributes** — age (as distance from the peak band, not linearly),
  height, the left-vs-right-hander matchup, ranking and ranking points in log
  space, and home-crowd advantage.

---

## Measured performance

Walk-forward: each season is predicted by a model trained **only on earlier
seasons**, which is the only defensible way to evaluate a sports model. A
cross-validated score on shuffled tennis data is meaningless, because ratings,
form and head-to-head all encode the future once the ordering is broken.

Run `make backtest` to reproduce the table; the dashboard's **Model card** tab
shows the same numbers from the trained bundle.

> **On the numbers in this repository.** The environment this was built in has no
> network access to the public archives, so the shipped artifacts are trained on
> the synthetic tour in `tennisdash/data/synthetic.py`. Those results demonstrate
> that the pipeline works end to end and that each component beats its baseline —
> they are **not** claims about real-tour accuracy. Run `make data && make train`
> on a machine with network access to get real numbers. See *Data sources* below.

What to look at, and why:

- **Log loss** and **Brier score** are proper scoring rules — minimised only by
  reporting your true belief — so they punish bad confidence as well as bad
  discrimination.
- **Calibration slope** should be 1.0. Below 1 means over-confident, which is the
  usual failure and the one that ruins a model in use.
- **Skill vs Elo** is the headline. Absolute log loss says nothing without knowing
  how hard the sample was; the fractional improvement over a pure Elo baseline on
  the identical rows is the number that says whether the extra machinery earns
  its keep.
- **Accuracy** is the least informative number and the one most often quoted.
  Predicting the higher-ranked player already gets ~65% right on tour, and a model
  can raise accuracy while getting *worse* at saying how confident it is.

---

## How it fits together

```
data/raw/*.csv                 public archives (or synthetic), untouched
        │  ingest.py           normalise, parse scorelines, attach venue context
        ▼
data/processed/matches         one row per match, winner/loser layout
        │
        ├── elo.py             4 rating families, chronological single pass
        ├── serve_return.py    ridge fit, refitted every 28 days on prior data only
        ├── history.py         form, fatigue, clutch, streaks, H2H
        │  builder.py          → antisymmetric p1/p2 feature matrix
        ▼
data/processed/features        75 features, labels balanced by construction
        │  ensemble.py         4 base learners → logistic stacker → isotonic calibration
        ▼
data/artifacts/model.joblib    ensemble + fitted engines + player directory + model card
        │  api/server.py       FastAPI
        ▼
web/                           dashboard
```

### Two invariants the code guarantees

**No look-ahead.** Every rating and history feature is produced by a single
chronological pass that reads state before writing it, and the serve/return model
is refitted on a fixed cadence using only `match_date < cut`. `tests/test_leakage.py`
checks this directly rather than trusting it: it re-runs the pipeline with results
flipped and asserts the pre-match features do not move, asserts a player's debut
match carries the default rating, asserts career counters lag by exactly one, and
corrupts every future serve stat to confirm a past fit does not notice.

**Exact antisymmetry.** `P(A beats B) + P(B beats A) == 1` holds to machine
precision, not approximately. Every feature is either a difference between the two
sides or a property of the match; the side assignment is randomised per match; and
at prediction time both orientations are scored and averaged. The test suite
asserts every column in the matrix either negates, stays fixed, or (for
`markov_prob`) maps to its complement under the swap.

There is exactly **one** feature builder, used for both training and serving.
Train/serve skew — where the served model quietly sees slightly different features
from the trained one — is a common way a good model becomes a bad product, and the
only reliable defence is not having a second implementation.

**No extrapolation at prediction time.** A hypothetical matchup is not drawn from
the same distribution as a real scheduled match: a user can pick two players who
last competed a year apart, which essentially never happens inside a live draw.
Left alone, the linear members extrapolate from that happily and badly — a query
like that could produce a prediction dominated by "days since last match" rather
than by either player's ability. Every live feature row is therefore clamped to
the 1st–99th percentile envelope the model was actually fitted on, and the bound
on an antisymmetric feature is deliberately symmetric so the clamp cannot break
the antisymmetry guarantee.

---

## The dashboard

- **Predictor** — win probability, projected hold rates and serve percentages,
  likely scorelines, and an attribution panel that neutralises each *category* of
  evidence in turn (ratings, serve/return, form, rest, clutch, H2H) and re-scores.
  Categories rather than single features, because the individual features are
  heavily correlated and attributing them one at a time splits one real effect
  into six identical-looking small ones.
- **Players** — Elo by surface, opponent-adjusted serve/return next to the raw
  percentages so you can see how much of a record was schedule, plus form,
  workload and shrunk clutch numbers.
- **Rankings** — Elo leaderboards, overall or by surface.
- **Model card** — the measured backtest: metrics by tour and surface, the
  calibration curve, season-by-season log loss against the Elo baseline, ensemble
  weights and permutation importance.

The dashboard is served from the same origin as the API, has no external requests
and no build step, and follows the viewer's light/dark preference.

---

## Data sources

| Source | Used for | Licence |
|---|---|---|
| [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) | ATP results **and per-match serve stats** | CC BY-NC-SA 4.0 |
| [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) | WTA results and serve stats | CC BY-NC-SA 4.0 |
| [tennis-data.co.uk](http://www.tennis-data.co.uk) | Closing odds, *benchmarking only* | free for personal use |

The Sackmann archives are the standard source for tennis modelling because they
carry per-match counting stats — aces, double faults, service points, first serves
in and won, break points faced and saved — not just results. Those counting stats
are what make serve/return skill estimation possible at all.

The model never trains on odds; that would be circular. They exist so the backtest
can answer "is this better than the market?", which is the only external yardstick
that cannot be gamed by choosing a convenient metric.

If `make fetch` returns HTTP 403 or 404, your network is blocking GitHub — run the
fetch from a machine with access, or use `make demo` for the offline path.

---

## CLI

```bash
tennisdash fetch --tours atp wta --start-year 2000   # download archives
tennisdash synth                                     # or generate offline data
tennisdash ingest                                    # normalise
tennisdash train --rebuild                           # features + backtest + model
tennisdash backtest                                  # walk-forward table
tennisdash predict "Alcaraz" "Sinner" --surface Clay --best-of 5 --tour atp
tennisdash rankings --tour wta --surface Grass
tennisdash serve                                     # dashboard
```

## Tuning on your own data

Every hyper-parameter lives in `tennisdash/config.py` and is recorded inside the
trained bundle. The values shipped were tuned by sweeping out-of-sample error on
the synthetic tour — **re-tune them on real data**, especially the serve/return
ridge penalty and window, which were selected by minimising out-of-sample serve
percentage RMSE.

## Layout

```
tennisdash/
  config.py            every tunable, in one place
  data/                sources, ingest, score parser, venue metadata, simulator
  features/            elo · serve_return · history · builder
  models/              markov · base_learners · ensemble
  backtest/            walkforward · metrics
  predict.py           live prediction, factor attribution
  train.py             builds the deployable bundle
  api/server.py        FastAPI
web/                   dashboard (no build step)
tests/                105 tests, including the leakage and antisymmetry suites
```

## Caveats

- Probabilities are probabilities. A 75% favourite loses one match in four, and
  that is the model working correctly, not failing.
- Retirements are predicted as wins for the player still standing; the model does
  not forecast in-match injury.
- Qualifying draws, doubles and juniors are out of scope.
- Betting return is reported as a model diagnostic. It is not advice.
