"""Turning posted odds into probabilities, and scoring a model against them.

Bookmakers do not publish probabilities; they publish prices that sum to more
than 100%. The excess is the margin (the "vig" or "overround"), and removing it
correctly is the difference between a useful benchmark and a misleading one.

Why the method matters. The obvious approach - divide each implied probability
by their sum - assumes the margin is spread *proportionally* across outcomes.
It is not. Bookmakers load more margin onto longshots than onto favourites, the
long-documented favourite-longshot bias, so proportional de-vigging
systematically under-states the favourite's true probability. In a sport as
favourite-heavy as tennis, where first-round Grand Slam prices routinely sit
near 1.05, that bias is large enough to change whether a model looks better or
worse than the market.

Four methods are implemented so the choice is explicit and testable:

``multiplicative``  the naive proportional scaling - included as a baseline
``additive``        splits the margin equally in probability space
``power``           solves for k with sum(pi^k) = 1
``shin``            models the margin as protection against better-informed
                    bettors, which reproduces the favourite-longshot pattern
                    rather than assuming it away

Shin is the default.

**A result worth knowing:** on a *two-outcome* market - which every tennis
match-winner market is - Shin's method and the additive method give identical
answers. This is exact, not approximate: the test suite checks it to machine
precision across a grid of prices and margins. So the real choice in tennis is
between proportional scaling on one hand and Shin/additive on the other, and
the two names are kept only because the equivalence stops holding the moment a
market has three or more outcomes (set betting, tournament winner).

The practical size of the choice: on a 1.05 favourite with a 4% margin,
proportional de-vigging says 91.3% and Shin says 93.1%. Nearly two points of
probability, all of it in the direction that matters most, because that is
exactly where tennis prices cluster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("shin", "multiplicative", "additive", "power")


def _as_pair(odds_a, odds_b) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a scalar or array pair to matching 1-D arrays.

    `atleast_1d` rather than `asarray`: a scalar would otherwise produce a
    0-dimensional array that cannot be indexed or masked, so every caller would
    need its own scalar special case.
    """
    a = np.atleast_1d(np.asarray(odds_a, dtype=float))
    b = np.atleast_1d(np.asarray(odds_b, dtype=float))
    a, b = np.broadcast_arrays(a, b)
    return np.array(a, dtype=float), np.array(b, dtype=float)


def overround(odds_a, odds_b) -> np.ndarray:
    """Total implied probability. 1.05 means a 5% bookmaker margin."""
    a, b = _as_pair(odds_a, odds_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / a + 1.0 / b


def _multiplicative(pi_a: np.ndarray, pi_b: np.ndarray) -> np.ndarray:
    total = pi_a + pi_b
    return np.where(total > 0, pi_a / total, 0.5)


def _additive(pi_a: np.ndarray, pi_b: np.ndarray) -> np.ndarray:
    """Remove an equal slice of the margin from each outcome."""
    excess = (pi_a + pi_b - 1.0) / 2.0
    return np.clip(pi_a - excess, 1e-6, 1 - 1e-6)


def _power(pi_a: np.ndarray, pi_b: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Solve k such that pi_a**k + pi_b**k == 1, by bisection on k."""
    low = np.ones_like(pi_a)
    high = np.full_like(pi_a, 6.0)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        total = pi_a ** mid + pi_b ** mid
        too_big = total > 1.0
        low = np.where(too_big, mid, low)
        high = np.where(too_big, high, mid)
    k = 0.5 * (low + high)
    result = pi_a ** k
    return np.clip(result, 1e-6, 1 - 1e-6)


def _shin(pi_a: np.ndarray, pi_b: np.ndarray, iterations: int = 80) -> np.ndarray:
    """Shin's method: back out the true probabilities under insider trading.

    Shin models the book as facing a proportion ``z`` of bettors who know the
    outcome, and prices defensively against them. Inverting that model gives

        p_i = ( sqrt(z^2 + 4(1-z) * pi_i^2 / PI) - z ) / (2(1-z))

    where PI is the overround. ``z`` is the single free parameter and is pinned
    down by requiring the recovered probabilities to sum to one. Because the sum
    falls monotonically in ``z``, bisection is exact and fast - no closed form
    needed, and no risk of transcribing one wrongly.
    """
    total = pi_a + pi_b
    safe_total = np.where(total > 0, total, 1.0)

    def probabilities(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.clip(z, 0.0, 0.999999)
        denominator = 2.0 * (1.0 - z)
        out = []
        for pi in (pi_a, pi_b):
            inner = z ** 2 + 4.0 * (1.0 - z) * (pi ** 2) / safe_total
            out.append((np.sqrt(np.maximum(inner, 0.0)) - z) / denominator)
        return out[0], out[1]

    low = np.zeros_like(pi_a)
    high = np.full_like(pi_a, 0.9999)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        p_a, p_b = probabilities(mid)
        # The recovered total decreases as z rises, so move the bracket
        # accordingly.
        too_big = (p_a + p_b) > 1.0
        low = np.where(too_big, mid, low)
        high = np.where(too_big, high, mid)

    p_a, _ = probabilities(0.5 * (low + high))
    return np.clip(p_a, 1e-6, 1 - 1e-6)


def implied_probability(odds_a, odds_b, method: str = "shin") -> np.ndarray:
    """Margin-free probability that outcome A wins, from a two-way price."""
    if method not in METHODS:
        raise ValueError(f"unknown de-vig method {method!r}; expected one of {METHODS}")
    a, b = _as_pair(odds_a, odds_b)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 1.0) & (b > 1.0)

    result = np.full(a.shape, np.nan)
    if not valid.any():
        return result

    pi_a = 1.0 / a[valid]
    pi_b = 1.0 / b[valid]
    solver = {
        "multiplicative": _multiplicative,
        "additive": _additive,
        "power": _power,
        "shin": _shin,
    }[method]
    result[valid] = solver(pi_a, pi_b)
    return result


@dataclass
class MarketView:
    """The market's own forecast for a set of matches."""

    probability: np.ndarray   # de-vigged P(player 1 wins)
    overround: np.ndarray     # bookmaker margin
    method: str

    @property
    def available(self) -> np.ndarray:
        return np.isfinite(self.probability)


def market_view(odds_a, odds_b, method: str = "shin") -> MarketView:
    return MarketView(
        probability=implied_probability(odds_a, odds_b, method=method),
        overround=overround(odds_a, odds_b),
        method=method,
    )


# --------------------------------------------------------------------- edges
def expected_value(probability, odds) -> np.ndarray:
    """Expected profit per unit staked at these odds, given this probability."""
    p = np.atleast_1d(np.asarray(probability, dtype=float))
    o = np.atleast_1d(np.asarray(odds, dtype=float))
    return p * o - 1.0


def kelly_fraction(probability, odds, cap: float = 1.0) -> np.ndarray:
    """Kelly stake as a fraction of bankroll, floored at zero.

    Full Kelly maximises long-run growth but only if the probability estimate is
    exactly right; it is brutally unforgiving of over-confidence. Callers should
    scale this down (a quarter is conventional) rather than betting it raw.
    """
    p = np.atleast_1d(np.asarray(probability, dtype=float))
    o = np.atleast_1d(np.asarray(odds, dtype=float))
    b = o - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        stake = (p * b - (1.0 - p)) / b
    return np.clip(np.nan_to_num(stake, nan=0.0), 0.0, cap)


def _logit(p) -> np.ndarray:
    p = np.clip(np.atleast_1d(np.asarray(p, dtype=float)), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def closing_line_value(model_probability, market_probability) -> np.ndarray:
    """Signed log-odds disagreement between the model and the closing price.

    Descriptive only. Note what this is *not*: it is tempting to score a model
    by how often it leans toward the eventual winner, but that metric is broken.
    A model that correctly judges the market to be over-pricing an 82% favourite
    at a true 78% leans *against* a player who then wins four times in five, and
    scores 20% on "leaned the right way" while being exactly right. Use
    :func:`encompassing_test` for the question that metric was trying to ask.
    """
    return _logit(model_probability) - _logit(market_probability)


def encompassing_test(y, model_probability, market_probability) -> dict:
    """Does the model carry information the closing line does not?

    Regresses the outcome on both forecasts in log-odds space:

        logit P(win) = a + b1 * logit(market) + b2 * logit(model)

    The market is a very strong forecast, so the honest question is never "is
    the model good?" but "does the model add anything *given* the market?".

    * ``b2`` near zero: the market already contains everything the model knows.
    * ``b2`` clearly positive: the model has genuine incremental information.
    * ``b1`` near zero with ``b2`` near one: the model encompasses the market -
      a strong claim that should be treated with suspicion until it survives
      out-of-sample on a lot of matches.

    This is the standard forecast-encompassing regression, and it is the right
    tool because it conditions on the market instead of competing with it.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=float)
    model_logit = _logit(model_probability)
    market_logit = _logit(market_probability)
    usable = np.isfinite(model_logit) & np.isfinite(market_logit) & np.isfinite(y)
    if usable.sum() < 200 or len(np.unique(y[usable])) < 2:
        return {"n": int(usable.sum()), "note": "not enough data"}

    from ..backtest.metrics import log_loss_safe

    target = y[usable]
    market_only = market_logit[usable].reshape(-1, 1)
    both = np.column_stack([market_logit[usable], model_logit[usable]])

    baseline = LogisticRegression(C=1e6, max_iter=2000).fit(market_only, target)
    combined = LogisticRegression(C=1e6, max_iter=2000).fit(both, target)
    market_coefficient, model_coefficient = combined.coef_[0]

    # The coefficients alone are not enough. When the model tracks the market
    # closely the two regressors are collinear, the split between them is
    # arbitrary, and a large "model" coefficient means nothing. The identified
    # quantity is how much *better the fit gets* when the model is added, so
    # that is what the verdict is based on.
    baseline_loss = log_loss_safe(target, baseline.predict_proba(market_only)[:, 1])
    combined_loss = log_loss_safe(target, combined.predict_proba(both)[:, 1])
    gain = baseline_loss - combined_loss

    # Likelihood-ratio test rather than an arbitrary cut-off. Adding *any*
    # regressor improves in-sample fit a little, so the bar has to be the
    # improvement a useless one would produce by chance: under the null,
    # 2 * n * gain is chi-squared with one degree of freedom.
    count = int(usable.sum())
    lr_statistic = 2.0 * count * gain
    try:
        from scipy.stats import chi2

        p_value = float(chi2.sf(max(lr_statistic, 0.0), df=1))
    except ImportError:  # scipy is optional
        p_value = float("nan")
    significant = lr_statistic > 3.841  # chi2(1) at the 5% level

    correlation = float(np.corrcoef(market_logit[usable], model_logit[usable])[0, 1])
    return {
        "n": count,
        "market_coefficient": float(market_coefficient),
        "model_coefficient": float(model_coefficient),
        "log_loss_gain_over_market": float(gain),
        "lr_statistic": float(lr_statistic),
        "p_value": p_value,
        "collinearity": correlation,
        "model_adds_information": bool(significant and gain > 0),
        "intercept": float(combined.intercept_[0]),
    }


def evaluate_against_market(
    y,
    model_probability,
    odds_a,
    odds_b,
    method: str = "shin",
    edge_threshold: float = 0.03,
    kelly_scale: float = 0.25,
) -> dict:
    """Compare a model with the closing line on the rows where odds exist.

    Reports both the statistical comparison (whose probabilities score better)
    and the practical one (would acting on the disagreement have made money).
    The two can disagree, and when they do the statistical answer is the more
    reliable of the pair - betting returns on a few thousand matches carry
    enormous variance.
    """
    from ..backtest.metrics import brier, log_loss_safe

    y = np.asarray(y, dtype=float)
    model = np.asarray(model_probability, dtype=float)
    view = market_view(odds_a, odds_b, method=method)
    usable = view.available & np.isfinite(model) & np.isfinite(y)
    n = int(usable.sum())
    if n == 0:
        return {"n": 0, "note": "no overlapping odds"}

    y_u, model_u, market_u = y[usable], model[usable], view.probability[usable]
    odds_a_u = np.asarray(odds_a, dtype=float)[usable]
    odds_b_u = np.asarray(odds_b, dtype=float)[usable]

    edge_a = model_u * odds_a_u - 1.0
    edge_b = (1 - model_u) * odds_b_u - 1.0
    back_a = edge_a > edge_threshold
    back_b = (edge_b > edge_threshold) & ~back_a

    stakes = np.zeros(n)
    stakes[back_a] = kelly_fraction(model_u[back_a], odds_a_u[back_a]) * kelly_scale
    stakes[back_b] = kelly_fraction(1 - model_u[back_b], odds_b_u[back_b]) * kelly_scale

    returns = np.zeros(n)
    returns[back_a] = np.where(
        y_u[back_a] == 1, stakes[back_a] * (odds_a_u[back_a] - 1), -stakes[back_a]
    )
    returns[back_b] = np.where(
        y_u[back_b] == 0, stakes[back_b] * (odds_b_u[back_b] - 1), -stakes[back_b]
    )
    staked = stakes.sum()

    clv = closing_line_value(model_u, market_u)
    encompassing = encompassing_test(y_u, model_u, market_u)

    return {
        "n": n,
        "method": method,
        "model_log_loss": log_loss_safe(y_u, model_u),
        "market_log_loss": log_loss_safe(y_u, market_u),
        "model_brier": brier(y_u, model_u),
        "market_brier": brier(y_u, market_u),
        "beats_market": bool(log_loss_safe(y_u, model_u) < log_loss_safe(y_u, market_u)),
        "mean_overround": float(np.nanmean(view.overround[usable])),
        "mean_abs_clv": float(np.mean(np.abs(clv))),
        "market_coefficient": encompassing.get("market_coefficient"),
        "model_coefficient": encompassing.get("model_coefficient"),
        "log_loss_gain_over_market": encompassing.get("log_loss_gain_over_market"),
        "encompassing_p_value": encompassing.get("p_value"),
        "market_model_collinearity": encompassing.get("collinearity"),
        "model_adds_information": encompassing.get("model_adds_information"),
        "bets": int((stakes > 0).sum()),
        "staked": float(staked),
        "profit": float(returns.sum()),
        "roi": float(returns.sum() / staked) if staked > 0 else float("nan"),
    }
