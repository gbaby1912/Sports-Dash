/* Sports-Dash dashboard.
   No build step and no external requests: the page is served from the same
   origin as the API, so there is no CORS surface and it works offline. */

const state = {
  tour: "atp",
  players: [],
  panel: "predictor",
  model: null,
  rankSurface: "",
  activeOnly: true,
  pending: 0,
};

const $ = (id) => document.getElementById(id);
const fmtPct = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`);
const fmtNum = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, params) {
  const url = new URL(path, window.location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
  });
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function notice(message, isError) {
  const element = $("global-notice");
  if (!message) { element.classList.add("hidden"); return; }
  element.textContent = message;
  element.className = `notice${isError ? " error" : ""}`;
}

/* ------------------------------------------------------------------ tooltip */
const tooltip = $("tooltip");
function bindTooltip(element, html) {
  element.addEventListener("mouseenter", (event) => {
    tooltip.innerHTML = html;
    tooltip.classList.add("on");
    moveTooltip(event);
  });
  element.addEventListener("mousemove", moveTooltip);
  element.addEventListener("mouseleave", () => tooltip.classList.remove("on"));
}
function moveTooltip(event) {
  const pad = 14;
  const rect = tooltip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

/* -------------------------------------------------------------- components */
function tile(label, value, sub) {
  return `<div class="tile"><div class="k">${esc(label)}</div>
    <div class="v">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}

/** A horizontal bar row growing from the left. */
function barRow(name, value, display, max, color, tip) {
  const width = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<div class="name">${esc(name)}</div>
    <div class="track"><div class="fill" style="left:0;width:${width}%;background:${color}"></div></div>
    <div class="val">${display}</div>`;
  if (tip) bindTooltip(row, tip);
  return row;
}

/** A diverging row: positive grows right of centre, negative grows left. */
function divergeRow(name, value, display, max, tip) {
  const half = max > 0 ? Math.min(50, (Math.abs(value) / max) * 50) : 0;
  const positive = value >= 0;
  const row = document.createElement("div");
  row.className = "row diverge";
  const left = positive ? 50 : 50 - half;
  row.innerHTML = `<div class="name">${esc(name)}</div>
    <div class="track"><div class="fill" style="left:${left}%;width:${half}%;
      background:${positive ? "var(--p1)" : "var(--p2)"}"></div></div>
    <div class="val">${display}</div>`;
  if (tip) bindTooltip(row, tip);
  return row;
}

/* ------------------------------------------------------------------ players */
async function loadPlayers() {
  state.players = await api("/api/players", { tour: state.tour, limit: 200, min_matches: 15 });
  const options = state.players
    .map((p) => `<option value="${p.player_id}">${esc(p.name)}${p.last_rank ? ` · #${p.last_rank}` : ""}</option>`)
    .join("");
  ["p1", "p2", "player-pick"].forEach((id) => { $(id).innerHTML = options; });
  if (state.players.length > 1) {
    $("p1").value = state.players[0].player_id;
    $("p2").value = state.players[1].player_id;
  }
}

/* --------------------------------------------------------------- predictor */
async function runPrediction() {
  const p1 = $("p1").value, p2 = $("p2").value;
  if (!p1 || !p2) return;
  if (p1 === p2) {
    $("prediction-body").classList.add("hidden");
    $("prediction-spinner").textContent = "Pick two different players.";
    $("prediction-spinner").classList.remove("hidden");
    return;
  }
  const token = ++state.pending;
  try {
    const data = await api("/api/predict", {
      tour: state.tour, p1, p2,
      surface: $("surface").value,
      best_of: $("best-of").value,
      level: $("level").value,
      round: $("round").value,
      tournament: $("tournament").value || "Neutral Court",
      indoor: $("indoor").checked ? "true" : "",
    });
    if (token !== state.pending) return;   // a newer request already landed
    renderPrediction(data);
    notice(null);
  } catch (error) {
    notice(`Prediction failed: ${error.message}`, true);
  }
}

function renderPrediction(data) {
  $("prediction-spinner").classList.add("hidden");
  $("prediction-body").classList.remove("hidden");

  const p1Name = data.p1.name, p2Name = data.p2.name;
  const p = data.p1_win_probability;
  $("hero-p1").lastElementChild.textContent = p1Name;
  $("hero-p2").lastElementChild.textContent = p2Name;
  $("legend-p1").textContent = p1Name;
  $("legend-p2").textContent = p2Name;

  const bar = $("probbar");
  bar.children[0].style.flexGrow = Math.max(p, 0.06);
  bar.children[1].style.flexGrow = Math.max(1 - p, 0.06);
  bar.children[0].textContent = fmtPct(p);
  bar.children[1].textContent = fmtPct(1 - p);
  bar.setAttribute("aria-label", `${p1Name} ${fmtPct(p)}, ${p2Name} ${fmtPct(1 - p)}`);

  const context = data.context;
  $("hero-context").textContent =
    `${context.surface}${context.indoor ? " · indoor" : " · outdoor"}` +
    `${context.altitude_m > 500 ? ` · ${context.altitude_m} m altitude` : ""}` +
    ` · best of ${context.best_of} · ${context.round} · ${context.tournament}`;

  renderShapeTiles(data);
  renderFactors(data);
  renderScores(data);
  renderServeChart(data);
  renderBaseModels(data);
  renderRatings(data);
}

function renderShapeTiles(data) {
  const s = data.serve || {};
  const eloGap = (data.ratings?.p1?.elo_surface ?? 0) - (data.ratings?.p2?.elo_surface ?? 0);
  const holdGap = (s.p1_hold_pct ?? 0) - (s.p2_hold_pct ?? 0);
  const decisive = (data.score_distribution || {});
  const straight = Object.entries(decisive)
    .filter(([k]) => { const [a, b] = k.split("-").map(Number); return Math.min(a, b) === 0; })
    .reduce((sum, [, v]) => sum + v, 0);

  $("shape-tiles").innerHTML = [
    tile("Surface Elo gap", `${eloGap >= 0 ? "+" : ""}${fmtNum(eloGap, 0)}`,
         `favours ${eloGap >= 0 ? esc(data.p1.name) : esc(data.p2.name)}`),
    tile("Expected serve pts won", `${fmtPct(s.p1_expected_spw)} / ${fmtPct(s.p2_expected_spw)}`,
         "opponent-adjusted, this surface"),
    tile("Projected hold rate", `${fmtPct(s.p1_hold_pct)} / ${fmtPct(s.p2_hold_pct)}`,
         `gap ${holdGap >= 0 ? "+" : ""}${fmtPct(holdGap)}`),
    tile("Point model says", fmtPct(s.markov_probability),
         "before ensemble correction"),
    tile("Straight sets", fmtPct(straight), "either player"),
  ].join("");
}

function renderFactors(data) {
  const container = $("factors");
  container.innerHTML = "";

  const groups = data.factor_groups || [];
  if (groups.length) {
    const max = Math.max(...groups.map((g) => Math.abs(g.contribution)));
    groups.forEach((g) => {
      const who = g.favours === "p1" ? data.p1.name : data.p2.name;
      container.appendChild(divergeRow(
        g.label,
        g.favours === "p1" ? Math.abs(g.contribution) : -Math.abs(g.contribution),
        `${g.contribution >= 0 ? "+" : "−"}${Math.abs(g.contribution).toFixed(2)}`,
        max,
        `<b>${esc(g.label)}</b><br>${esc(g.description)}<br><br>
         Favours ${esc(who)} by ${Math.abs(g.contribution).toFixed(3)} in log-odds<br>
         <span style="opacity:.7">${g.n_features} feature${g.n_features === 1 ? "" : "s"} neutralised together</span>`
      ));
    });
  }

  const factors = (data.factors || []).slice(0, 6);
  if (factors.length) {
    const detail = document.createElement("details");
    detail.style.marginTop = "14px";
    const max = Math.max(...factors.map((f) => Math.abs(f.contribution)));
    detail.innerHTML = `<summary class="muted" style="cursor:pointer">Individual features</summary>
      <p class="muted" style="margin:10px 0 10px">
        One at a time. Because several of these move together, each looks smaller
        here than the category it belongs to — that is the correlation, not a
        disagreement.</p>
      <div class="rows" id="factor-detail"></div>`;
    container.appendChild(detail);
    const rows = detail.querySelector("#factor-detail");
    factors.forEach((f) => {
      const who = f.favours === "p1" ? data.p1.name : data.p2.name;
      rows.appendChild(divergeRow(
        f.label,
        f.favours === "p1" ? Math.abs(f.contribution) : -Math.abs(f.contribution),
        `${f.contribution >= 0 ? "+" : "−"}${Math.abs(f.contribution).toFixed(2)}`,
        max,
        `<b>${esc(f.label)}</b><br>Favours ${esc(who)}<br>
         raw feature value ${Number(f.value).toFixed(3)}`
      ));
    });
  }

  if (!groups.length && !factors.length) {
    container.innerHTML = `<p class="muted">These two are level on every measured factor.</p>`;
  }
}

function renderScores(data) {
  const container = $("scores");
  container.innerHTML = "";
  const entries = Object.entries(data.score_distribution || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { container.innerHTML = `<p class="muted">No score model available.</p>`; return; }
  const max = entries[0][1];
  entries.forEach(([score, probability]) => {
    const [a, b] = score.split("-").map(Number);
    const p1Wins = a > b;
    const winner = p1Wins ? data.p1.name : data.p2.name;
    container.appendChild(barRow(
      `${winner} ${p1Wins ? score : score.split("-").reverse().join("-")}`,
      probability, fmtPct(probability), max,
      p1Wins ? "var(--p1)" : "var(--p2)",
      `<b>${esc(winner)} in ${Math.max(a, b)}–${Math.min(a, b)} sets</b><br>${fmtPct(probability, 1)} of outcomes`
    ));
  });
}

function renderServeChart(data) {
  const s = data.serve || {};
  const rows = [
    { label: "Serve points won (adj.)", p1: s.p1_expected_spw, p2: s.p2_expected_spw, pct: true },
    { label: "Serve points won (raw)", p1: s.p1_raw_spw, p2: s.p2_raw_spw, pct: true },
    { label: "Return points won (raw)", p1: s.p1_raw_rpw, p2: s.p2_raw_rpw, pct: true },
    { label: "Hold probability", p1: s.p1_hold_pct, p2: s.p2_hold_pct, pct: true },
  ];
  const width = 500, rowHeight = 44, pad = { left: 8, right: 44, top: 26, bottom: 22 };
  const height = pad.top + rows.length * rowHeight + pad.bottom;
  const barWidth = width - pad.left - pad.right - 130;
  const x0 = pad.left + 130;
  const domain = [0.25, 0.95];
  const scale = (v) => ((v - domain[0]) / (domain[1] - domain[0])) * barWidth;

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img"
    aria-label="Serve and return comparison">`;
  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].forEach((t) => {
    const x = x0 + scale(t);
    svg += `<line class="grid-line" x1="${x}" y1="${pad.top - 6}" x2="${x}" y2="${height - pad.bottom}"/>
            <text class="tick" x="${x}" y="${height - pad.bottom + 13}" text-anchor="middle">${Math.round(t * 100)}%</text>`;
  });

  rows.forEach((row, index) => {
    const y = pad.top + index * rowHeight;
    svg += `<text class="direct-label" x="${pad.left}" y="${y + 15}">${esc(row.label)}</text>`;
    [["p1", "var(--p1)"], ["p2", "var(--p2)"]].forEach(([key, color], seriesIndex) => {
      const value = row[key];
      if (value === null || value === undefined) return;
      const w = Math.max(scale(value), 2);
      const by = y + 20 + seriesIndex * 9;
      svg += `<rect class="mark" x="${x0}" y="${by}" width="${w}" height="7" rx="3.5" fill="${color}"
                data-tip="${esc(row.label)}|${key === "p1" ? esc(data.p1.name) : esc(data.p2.name)}|${fmtPct(value)}"/>`;
      svg += `<text class="direct-label" x="${x0 + w + 5}" y="${by + 7}">${fmtPct(value, 1)}</text>`;
    });
  });
  svg += `</svg>`;
  $("serve-chart").innerHTML =
    `<div class="legend">
       <span><span class="swatch" style="background:var(--p1)"></span>${esc(data.p1.name)}</span>
       <span><span class="swatch" style="background:var(--p2)"></span>${esc(data.p2.name)}</span>
     </div>${svg}`;

  $("serve-chart").querySelectorAll("[data-tip]").forEach((mark) => {
    const [metric, who, value] = mark.dataset.tip.split("|");
    bindTooltip(mark, `<b>${who}</b><br>${metric}: ${value}`);
  });
}

function renderBaseModels(data) {
  const container = $("base-models");
  container.innerHTML = "";
  const labels = {
    rating: "Elo ratings",
    markov: "Point-by-point model",
    gbm: "Gradient boosting",
    linear: "Regularised logistic",
  };
  const notes = {
    rating: "Who has been beating whom, weighted by surface and margin.",
    markov: "Serve and return skill run through the scoring system.",
    gbm: "Non-linear interactions across all 75 features.",
    linear: "A stable linear read on the same features.",
  };
  const entries = Object.entries(data.base_models || {});
  entries.forEach(([name, probability]) => {
    container.appendChild(barRow(
      labels[name] || name, probability, fmtPct(probability), 1.0,
      probability >= 0.5 ? "var(--p1)" : "var(--p2)",
      `<b>${labels[name] || name}</b><br>${notes[name] || ""}<br>
       ${esc(data.p1.name)} ${fmtPct(probability)}`
    ));
  });
  const spread = entries.length
    ? Math.max(...entries.map(([, v]) => v)) - Math.min(...entries.map(([, v]) => v)) : 0;
  container.appendChild(Object.assign(document.createElement("p"), {
    className: "muted",
    style: "margin:10px 0 0",
    textContent: `Spread across models: ${fmtPct(spread)}${spread > 0.12
      ? " — the members disagree, treat this one as genuinely uncertain."
      : " — the members broadly agree."}`,
  }));
}

function renderRatings(data) {
  const r = data.ratings || {};
  const rows = [
    ["Overall Elo", r.p1?.elo, r.p2?.elo, 0],
    ["Surface Elo", r.p1?.elo_surface, r.p2?.elo_surface, 0],
    ["Points-won Elo", r.p1?.elo_points, r.p2?.elo_points, 0],
    ["Career peak Elo", r.p1?.peak_elo, r.p2?.peak_elo, 0],
    ["Rated matches", r.p1?.matches, r.p2?.matches, 0],
  ];
  $("ratings-table").innerHTML = `<table>
    <thead><tr><th>Rating</th><th>${esc(data.p1.name)}</th><th>${esc(data.p2.name)}</th></tr></thead>
    <tbody>${rows.map(([label, a, b, d]) =>
      `<tr><td>${label}</td><td>${fmtNum(a, d)}</td><td>${fmtNum(b, d)}</td></tr>`).join("")}
    </tbody></table>`;
}

/* ----------------------------------------------------------------- profile */
async function loadProfile() {
  const playerId = $("player-pick").value;
  if (!playerId) return;
  try {
    const p = await api(`/api/player/${state.tour}/${playerId}`, { surface: $("surface").value });
    renderProfile(p);
  } catch (error) {
    $("profile-body").innerHTML = `<div class="notice error">${esc(error.message)}</div>`;
  }
}

function renderProfile(p) {
  const elo = p.elo || {}, sr = p.serve_return || {}, form = p.form || {},
        fatigue = p.fatigue || {}, clutch = p.clutch || {}, record = p.record || {};
  const surfaces = ["hard", "clay", "grass"];

  // Elo bars are measured from 1500 (the tour average a new player starts at),
  // not from zero. Scaling from zero would render 2036 and 2223 as bars 92% and
  // 100% long - visually identical, when the gap is actually a large one.
  const BASE_ELO = 1500;
  const eloMax = Math.max(...surfaces.map((s) => elo[s] || 0), elo.overall || 0, BASE_ELO + 1);
  const eloSpan = Math.max(eloMax - BASE_ELO, 1);
  const surfaceRows = surfaces.map((s) => {
    const value = elo[s] || 0;
    const width = Math.max(((value - BASE_ELO) / eloSpan) * 100, 1.5);
    return `<div class="row"><div class="name">${s[0].toUpperCase() + s.slice(1)}</div>
      <div class="track"><div class="fill" style="left:0;width:${width}%;
        background:var(--seq-450)"></div></div>
      <div class="val">${fmtNum(value, 0)}</div></div>`;
  }).join("");

  const byS = sr.by_surface || {};
  const rated = sr.rated !== false;
  const srRows = surfaces.map((s) => {
    const d = byS[s] || {};
    return `<tr><td>${s[0].toUpperCase() + s.slice(1)}</td>
      <td>${rated ? (d.serve_skill >= 0 ? "+" : "") + fmtNum(d.serve_skill, 3) : "–"}</td>
      <td>${rated ? (d.return_skill >= 0 ? "+" : "") + fmtNum(d.return_skill, 3) : "–"}</td>
      <td>${rated ? fmtPct(d.neutral_spw) : "–"}</td>
      <td>${rated ? fmtPct(d.neutral_rpw) : "–"}</td></tr>`;
  }).join("");

  $("profile-body").innerHTML = `
    <div class="card">
      <div class="card-head">
        <h2>${esc(p.name)}</h2>
        <span class="pill">${esc(p.ioc || "—")}</span>
        <span class="pill">${p.hand === "L" ? "Left-handed" : p.hand === "R" ? "Right-handed" : "Hand unknown"}</span>
        ${p.height_cm ? `<span class="pill">${fmtNum(p.height_cm, 0)} cm</span>` : ""}
        <span class="card-note">Rank #${fmtNum(p.rank, 0)} · ${p.matches} matches on record${
          p.as_of ? ` · as of ${String(p.as_of).slice(0, 10)}` : ""}</span>
      </div>
      <div class="tiles">
        ${tile("Overall Elo", fmtNum(elo.overall, 0), `peak ${fmtNum(elo.peak, 0)}`)}
        ${tile("Career win rate", fmtPct(p.win_pct), `${record.career_wins || 0} wins`)}
        ${tile("Recent form", fmtPct(form.form_decayed), "time-decayed win rate")}
        ${tile("Serve skill", `${(sr.overall?.serve_skill ?? 0) >= 0 ? "+" : ""}${fmtNum(sr.overall?.serve_skill, 3)}`,
               `raw ${fmtPct(sr.overall?.raw_spw)} of serve points`)}
        ${tile("Return skill", `${(sr.overall?.return_skill ?? 0) >= 0 ? "+" : ""}${fmtNum(sr.overall?.return_skill, 3)}`,
               `raw ${fmtPct(sr.overall?.raw_rpw)} of return points`)}
        ${tile("Current streak",
               record.win_streak > 0 ? `W${record.win_streak}` : record.loss_streak > 0 ? `L${record.loss_streak}` : "–",
               "consecutive results")}
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>Elo by surface</h3></div>
        <p class="muted" style="margin:-6px 0 12px">
          Blended toward overall Elo where the surface sample is thin. Bars are
          measured from 1500, the rating a new player starts at.
        </p>
        <div class="rows">${surfaceRows}</div>
      </div>

      <div class="card">
        <div class="card-head"><h3>Opponent-adjusted serve &amp; return</h3></div>
        <p class="muted" style="margin:-6px 0 12px">
          In log-odds of winning a point. Positive is above tour average, purged of
          who this player happened to face.
        </p>
        <div class="table-wrap"><table>
          <thead><tr><th>Surface</th><th>Serve</th><th>Return</th>
            <th>SPW vs avg</th><th>RPW vs avg</th></tr></thead>
          <tbody>${srRows}</tbody></table></div>
        <p class="muted" style="margin-top:10px">
          The last two columns translate the skill numbers back into percentages:
          what this player would win against a league-average opponent, which is
          what a raw career percentage is trying and failing to measure.
          Based on ${fmtNum(sr.overall?.service_points, 0)} weighted service points.
        </p>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>Clutch</h3><span class="card-note">shrunk toward tour average</span></div>
        <p class="muted" style="margin:-6px 0 12px">
          Single-season clutch splits are mostly noise, so these are pulled toward
          the tour mean in proportion to how little evidence there is.
        </p>
        <div class="tiles">
          ${tile("Break points saved", fmtPct(clutch.bp_saved_pct))}
          ${tile("Break points converted", fmtPct(clutch.bp_conv_pct))}
          ${tile("Tiebreaks won", fmtPct(clutch.tiebreak_pct))}
          ${tile("Deciding sets won", fmtPct(clutch.decider_pct))}
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h3>Workload</h3>
          <span class="card-note">at the end of the data</span></div>
        <div class="tiles">
          ${tile("Days since last match", fmtNum(fatigue.days_since_last, 0))}
          ${tile("Matches, 14 days", fmtNum(fatigue.matches_14d, 0))}
          ${tile("Court time, 28 days", `${fmtNum(fatigue.minutes_28d, 0)} min`)}
          ${tile("Retirements, 90 days", fmtNum(fatigue.recent_retirements, 0), "injury proxy")}
        </div>
      </div>
    </div>`;
}

/* ---------------------------------------------------------------- rankings */
async function loadRankings() {
  try {
    const rows = await api("/api/rankings", {
      tour: state.tour, surface: state.rankSurface, top: 40, min_matches: 20,
      active_days: state.activeOnly ? 365 : 0,
    });
    renderRankings(rows);
  } catch (error) {
    $("rankings-body").innerHTML = `<div class="notice error">${esc(error.message)}</div>`;
  }
}

/** Serve/return skill, or a dash when the player has no rating in the current
    window. Printing 0.000 there would claim they are exactly tour average. */
function skillCell(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return `<span class="muted" title="No matches in the rating window">–</span>`;
  }
  return `${value >= 0 ? "+" : ""}${fmtNum(value, 3)}`;
}

function renderRankings(rows) {
  if (!rows.length) { $("rankings-body").innerHTML = `<p class="muted">No rated players yet.</p>`; return; }
  const key = state.rankSurface ? `elo_${state.rankSurface.toLowerCase()}` : "elo";
  const max = Math.max(...rows.map((r) => r[key] || 0));
  const min = Math.min(...rows.map((r) => r[key] || 0));
  const body = rows.map((r, i) => {
    const value = r[key] || 0;
    const width = max > min ? ((value - min) / (max - min)) * 100 : 100;
    return `<tr>
      <td>${i + 1}. ${esc(r.name || r.player_id)} <span class="muted">${esc(r.ioc || "")}</span></td>
      <td style="width:38%">
        <div class="track" style="position:relative;height:13px;background:var(--grid);border-radius:4px">
          <div style="position:absolute;left:0;top:0;bottom:0;width:${Math.max(width, 2)}%;
               background:var(--seq-450);border-radius:4px"></div></div></td>
      <td>${fmtNum(value, 0)}</td>
      <td>${fmtNum(r.peak_elo, 0)}</td>
      <td>${skillCell(r.serve_skill)}</td>
      <td>${skillCell(r.return_skill)}</td>
      <td>${fmtNum(r.matches, 0)}</td></tr>`;
  }).join("");

  $("rankings-body").innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Player</th><th>${state.rankSurface || "Overall"} Elo</th><th></th>
      <th>Peak</th><th>Serve</th><th>Return</th><th>Matches</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

/* -------------------------------------------------------------- model card */
async function loadModelCard() {
  try {
    state.model = state.model || await api("/api/model");
    renderModelCard(state.model);
  } catch (error) {
    $("model-body").innerHTML = `<div class="notice error">${esc(error.message)}</div>`;
  }
}

function renderModelCard(meta) {
  const report = meta.report || {};
  const pooled = report.backtest_pooled || [];
  const overall = pooled.find((r) => r.group === "all");
  const seasons = (report.backtest_by_season || []).filter((r) => r.tour === "all");

  const headline = overall ? `
    <div class="tiles">
      ${tile("Log loss", overall.log_loss.toFixed(4),
             `Elo baseline ${overall.ref_elo_log_loss?.toFixed(4) ?? "–"}`)}
      ${tile("Skill vs Elo", `${(overall.skill_vs_elo * 100).toFixed(1)}%`, "log-loss reduction")}
      ${tile("Accuracy", fmtPct(overall.accuracy, 1), `${overall.n.toLocaleString()} matches`)}
      ${tile("Brier score", overall.brier.toFixed(4), "lower is better")}
      ${tile("Calibration error", overall.ece.toFixed(4), "mean |predicted − observed|")}
      ${tile("Calibration slope", overall.calibration_slope.toFixed(3),
             overall.calibration_slope < 0.95 ? "slightly over-confident"
             : overall.calibration_slope > 1.05 ? "slightly under-confident" : "well calibrated")}
    </div>` : `<p class="muted">No backtest recorded in this bundle.</p>`;

  const groupRows = pooled.map((r) => `<tr>
      <td>${esc(r.group)}</td><td>${r.n.toLocaleString()}</td>
      <td>${r.log_loss.toFixed(4)}</td><td>${r.ref_elo_log_loss?.toFixed(4) ?? "–"}</td>
      <td>${fmtPct(r.accuracy, 1)}</td><td>${r.ece.toFixed(4)}</td>
      <td>${(r.skill_vs_elo * 100).toFixed(1)}%</td></tr>`).join("");

  const calibration = report.calibration || [];
  const importance = (report.feature_importance || []).slice(0, 18);
  const impMax = importance.length ? Math.max(...importance.map((f) => f.importance)) : 1;
  const impRows = importance.map((f) => `<div class="row">
      <div class="name">${esc(f.feature)}</div>
      <div class="track"><div class="fill" style="left:0;
        width:${Math.max((f.importance / impMax) * 100, 1)}%;background:var(--seq-450)"></div></div>
      <div class="val">${f.importance.toFixed(4)}</div></div>`).join("");

  const weights = meta.stacker_weights || {};
  const weightRows = Object.entries(weights)
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.toFixed(3)}</td></tr>`).join("");

  $("model-body").innerHTML = `
    <div class="card">
      <div class="card-head">
        <h2>Measured out-of-sample performance</h2>
        <span class="card-note">walk-forward: each season predicted by a model that saw only earlier seasons</span>
      </div>
      ${headline}
      <p class="muted" style="margin-top:14px">
        Accuracy is the least informative number here. A model can raise accuracy while
        getting worse at saying how confident it is — log loss, Brier score and the
        calibration slope are the ones that catch that.
      </p>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>By tour and surface</h3></div>
        <div class="table-wrap"><table>
          <thead><tr><th>Group</th><th>N</th><th>Log loss</th><th>Elo</th>
            <th>Acc</th><th>ECE</th><th>Skill</th></tr></thead>
          <tbody>${groupRows}</tbody></table></div>
      </div>
      <div class="card">
        <div class="card-head"><h3>Calibration</h3><span class="card-note">predicted vs observed</span></div>
        ${calibrationChart(calibration)}
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>Season by season</h3></div>
        ${seasonChart(seasons)}
      </div>
      <div class="card">
        <div class="card-head"><h3>Ensemble weights</h3></div>
        <p class="muted" style="margin:-6px 0 12px">
          Stacker coefficients on each member's log-odds, fitted out-of-sample. A
          weight can be negative: the members share features, so the stacker uses
          one to correct another's double-counting rather than to overrule it.
        </p>
        <div class="table-wrap"><table>
          <thead><tr><th>Member</th><th>Weight</th></tr></thead>
          <tbody>${weightRows}</tbody></table></div>
        <p class="muted" style="margin-top:14px">
          Trained on ${Number(meta.training_rows).toLocaleString()} matches
          (${esc(String(meta.data_span?.[0] ?? "").slice(0, 10))} to
           ${esc(String(meta.data_span?.[1] ?? "").slice(0, 10))}),
          ${meta.n_features} features, ${esc(meta.calibration_method || "")} calibration.
        </p>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>Feature importance</h3>
        <span class="card-note">permutation, measured on the ensemble</span>
      </div>
      <p class="muted" style="margin:-6px 0 12px">
        Increase in log loss when the feature is shuffled. Correlated features share
        credit, so a low score here means "redundant", not "irrelevant".
      </p>
      <div class="rows">${impRows || '<p class="muted">Not computed for this bundle.</p>'}</div>
    </div>`;

  bindChartTooltips();
}

function calibrationChart(bins) {
  if (!bins.length) return `<p class="muted">No calibration data.</p>`;
  const width = 380, height = 300, pad = { l: 44, r: 14, t: 12, b: 38 };
  const iw = width - pad.l - pad.r, ih = height - pad.t - pad.b;
  const sx = (v) => pad.l + v * iw;
  const sy = (v) => pad.t + (1 - v) * ih;

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img"
    aria-label="Calibration: predicted versus observed win rate">`;
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    svg += `<line class="grid-line" x1="${pad.l}" y1="${sy(t)}" x2="${width - pad.r}" y2="${sy(t)}"/>
            <text class="tick" x="${pad.l - 7}" y="${sy(t) + 4}" text-anchor="end">${Math.round(t * 100)}%</text>
            <text class="tick" x="${sx(t)}" y="${height - pad.b + 15}" text-anchor="middle">${Math.round(t * 100)}%</text>`;
  });
  // The reference line: perfect calibration sits exactly on the diagonal.
  svg += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"
           stroke="var(--baseline)" stroke-width="1" stroke-dasharray="4 3"/>`;
  const path = bins.map((b, i) => `${i ? "L" : "M"}${sx(b.predicted)},${sy(b.observed)}`).join(" ");
  svg += `<path d="${path}" fill="none" stroke="var(--p1)" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round"/>`;
  bins.forEach((b) => {
    svg += `<circle class="mark" cx="${sx(b.predicted)}" cy="${sy(b.observed)}" r="4.5"
             fill="var(--p1)" stroke="var(--surface-1)" stroke-width="2"
             data-tip="Predicted ${fmtPct(b.predicted)}|Observed ${fmtPct(b.observed)}|${b.n.toLocaleString()} matches"/>`;
  });
  svg += `<text class="axis-label" x="${pad.l + iw / 2}" y="${height - 4}" text-anchor="middle">Predicted win probability</text>
          <text class="axis-label" transform="rotate(-90 12 ${pad.t + ih / 2})" x="12" y="${pad.t + ih / 2}"
            text-anchor="middle">Observed win rate</text></svg>`;
  return `<div class="chart">${svg}</div>
    <p class="muted" style="margin-top:8px">The dashed diagonal is perfect calibration.</p>`;
}

function seasonChart(seasons) {
  if (!seasons.length) return `<p class="muted">No seasonal data.</p>`;
  const width = 380, height = 300, pad = { l: 48, r: 14, t: 12, b: 38 };
  const iw = width - pad.l - pad.r, ih = height - pad.t - pad.b;
  const all = seasons.flatMap((s) => [s.log_loss, s.elo_log_loss]);
  const lo = Math.min(...all) - 0.01, hi = Math.max(...all) + 0.01;
  const sx = (i) => pad.l + (seasons.length > 1 ? (i / (seasons.length - 1)) * iw : iw / 2);
  const sy = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * ih;

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img"
    aria-label="Log loss by season, model versus Elo baseline">`;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo);
    svg += `<line class="grid-line" x1="${pad.l}" y1="${sy(v)}" x2="${width - pad.r}" y2="${sy(v)}"/>
            <text class="tick" x="${pad.l - 7}" y="${sy(v) + 4}" text-anchor="end">${v.toFixed(3)}</text>`;
  }
  // Label every k-th season, and only include the final year if it will not
  // collide with the previous label.
  const step = Math.ceil(seasons.length / 6);
  const ticks = seasons.map((_, i) => i).filter((i) => i % step === 0);
  const last = seasons.length - 1;
  if (last - ticks[ticks.length - 1] >= Math.max(step - 1, 1)) ticks.push(last);
  ticks.forEach((i) => {
    svg += `<text class="tick" x="${sx(i)}" y="${height - pad.b + 15}" text-anchor="middle">${seasons[i].year}</text>`;
  });
  const line = (key, color) =>
    `<path d="${seasons.map((s, i) => `${i ? "L" : "M"}${sx(i)},${sy(s[key])}`).join(" ")}"
      fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
  svg += line("elo_log_loss", "var(--p2)") + line("log_loss", "var(--p1)");
  seasons.forEach((s, i) => {
    svg += `<circle class="mark" cx="${sx(i)}" cy="${sy(s.log_loss)}" r="4" fill="var(--p1)"
             stroke="var(--surface-1)" stroke-width="2"
             data-tip="${s.year}|Model ${s.log_loss.toFixed(4)}|Elo ${s.elo_log_loss.toFixed(4)}"/>`;
  });
  svg += `</svg>`;
  return `<div class="legend">
      <span><span class="swatch" style="background:var(--p1)"></span>This model</span>
      <span><span class="swatch" style="background:var(--p2)"></span>Elo baseline</span>
    </div><div class="chart">${svg}</div>
    <p class="muted" style="margin-top:8px">Lower is better. Both lines are out-of-sample.</p>`;
}

function bindChartTooltips() {
  document.querySelectorAll("#model-body [data-tip]").forEach((mark) => {
    const parts = mark.dataset.tip.split("|");
    bindTooltip(mark, `<b>${parts[0]}</b><br>${parts.slice(1).join("<br>")}`);
  });
}

/* -------------------------------------------------------------------- wire */
function showPanel(name) {
  state.panel = name;
  document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
  $(`panel-${name}`).classList.remove("hidden");
  document.querySelectorAll(".tab").forEach((t) =>
    t.setAttribute("aria-selected", String(t.dataset.panel === name)));
  if (name === "rankings") loadRankings();
  if (name === "model") loadModelCard();
  if (name === "players") loadProfile();
}

async function switchTour(tour) {
  state.tour = tour;
  $("tour-atp").setAttribute("aria-pressed", String(tour === "atp"));
  $("tour-wta").setAttribute("aria-pressed", String(tour === "wta"));
  await loadPlayers();
  if (state.panel === "predictor") runPrediction();
  if (state.panel === "rankings") loadRankings();
  if (state.panel === "players") loadProfile();
}

function wire() {
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => showPanel(tab.dataset.panel)));
  $("tour-atp").addEventListener("click", () => switchTour("atp"));
  $("tour-wta").addEventListener("click", () => switchTour("wta"));

  ["p1", "p2", "surface", "best-of", "round", "level", "indoor"].forEach((id) =>
    $(id).addEventListener("change", runPrediction));
  let debounce;
  $("tournament").addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(runPrediction, 350);
  });

  $("player-search").addEventListener("input", (event) => {
    const query = event.target.value.toLowerCase();
    const matched = state.players.filter((p) => p.name.toLowerCase().includes(query));
    $("player-pick").innerHTML = matched
      .map((p) => `<option value="${p.player_id}">${esc(p.name)}</option>`).join("");
    if (matched.length) loadProfile();
  });
  $("player-pick").addEventListener("change", loadProfile);

  $("active-only").addEventListener("change", (event) => {
    state.activeOnly = event.target.checked;
    loadRankings();
  });

  $("rank-surface").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    state.rankSurface = button.dataset.surface;
    $("rank-surface").querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b === button)));
    loadRankings();
  });
}

async function init() {
  wire();
  try {
    const health = await api("/api/health");
    if (!health.model_loaded) {
      notice("No trained model found. Run `make all` (or `tennisdash train`) and reload.", true);
      return;
    }
    await loadPlayers();
    state.model = await api("/api/model");
    const span = state.model.data_span || [];
    $("footer").textContent =
      `Model trained on ${Number(state.model.training_rows).toLocaleString()} matches ` +
      `(${String(span[0] ?? "").slice(0, 10)} – ${String(span[1] ?? "").slice(0, 10)}), ` +
      `${state.model.n_features} features. Predictions are probabilities, not certainties.`;
    // Seed the tournament suggestions from the venues the model knows about.
    await runPrediction();
  } catch (error) {
    notice(`Could not reach the API: ${error.message}`, true);
  }
}

init();
