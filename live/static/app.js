// Live training dashboard. Polls the Python server (live/server.py) and draws:
//  - the skill-check curve (same words, best guesses: what the AI has learned)
//  - the practice games it learns from (random words, some exploring)
//  - its weights, what it learned to do (probing), the openers it tried,
//  - and the word-difficulty analysis / lookup.
// All the learning happens in Python (live/session.py).

const SPEEDS = [1, 2, 5, 10, 25, 50, 100, 0]; // games per second; 0 = as fast as possible
const TARGETS = [100, 250, 500, 1000, 2000, 5000];
const POLL_MS = 250;
const EXAMPLES = { 3: "cat", 4: "rope", 5: "foyer", 6: "planet", 7: "kitchen", 8: "elephant", 9: "adventure",
  10: "background" };
let wordsName = null;
const WEIGHT_LABELS = {
  distinct_letters: "Different letters in the word",
  possible_letters: "Common letters (among possible words)",
  possible_positions: "Letters in their common spots",
  win_chance: "Chance it's the answer (go for the win)",
  avg_left: "Words left afterwards, on average",
  worst_left: "Words left in the worst case",
  patterns: "Different color patterns it can produce",
};

let runId = null;
let windowSize = 50;
const games = [];       // practice games: {x, y, secret}
const averages = [];    // rolling average of practice games: {x, y}
const skillPoints = []; // skill checks: {x: games trained, y: average guesses}
let windowSum = 0;
let dragging = { speed: false, target: false, length: false };
let analysisKey = null;
let skillChart = null;
let practiceChart = null;

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

// ---------------------------------------------------------------- charts

const crosshair = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const active = chart.tooltip && chart.tooltip.getActiveElements();
    if (!active || !active.length) return;
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = css("--axis");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(active[0].element.x, top);
    ctx.lineTo(active[0].element.x, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function baseOptions(targetGames, xTitle, yTitle) {
  return {
    animation: false,
    locale: "en-US", // same number format as the tiles
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { type: "linear", min: 0, max: targetGames, ticks: {}, title: { display: true, text: xTitle } },
      y: { ticks: {}, title: { display: true, text: yTitle } },
    },
    plugins: { legend: { display: false }, tooltip: { displayColors: false } },
  };
}

function makeCharts(targetGames) {
  if (!window.Chart) {
    document.querySelectorAll(".chart-error").forEach((node) => { node.hidden = false; });
    return;
  }
  const skillOptions = baseOptions(targetGames, "games trained", "average guesses");
  skillOptions.scales.y.suggestedMin = 3.3;
  skillOptions.scales.y.suggestedMax = 4.4;
  skillOptions.plugins.tooltip.filter = (item) => item.datasetIndex === 0;
  skillOptions.plugins.tooltip.callbacks = {
    title: (items) => items[0].raw.x === 0 ? "Game 0: blank AI, no training yet"
                                           : `After ${items[0].raw.x.toLocaleString("en-US")} games of training`,
    label: (item) => `${item.raw.y.toFixed(2)} average guesses`,
  };
  skillChart = new Chart($("skill-chart"), {
    data: {
      datasets: [
        { type: "line", label: "Skill check", data: skillPoints, borderWidth: 2, pointRadius: 4,
          pointHoverRadius: 6, pointBorderWidth: 2, tension: 0, borderJoinStyle: "round", order: 1 },
        { type: "line", label: "Blank AI level", data: [], borderWidth: 1, pointRadius: 0, pointHoverRadius: 0,
          order: 2 },
      ],
    },
    options: skillOptions,
    plugins: [crosshair],
  });

  const practiceOptions = baseOptions(targetGames, "practice games played", "guesses");
  practiceOptions.scales.y.min = 1;
  practiceOptions.scales.y.suggestedMax = 8;
  practiceOptions.scales.y.ticks = { stepSize: 1, precision: 0 };
  practiceOptions.plugins.tooltip.callbacks = {
    title: (items) => `Game ${items[0].raw.x}`,
    label: (item) => item.datasetIndex === 0
      ? `${item.raw.y} guesses · ${item.raw.secret.toUpperCase()}`
      : `${item.raw.y.toFixed(2)} average of the last ${Math.min(windowSize, item.raw.x)}`,
  };
  practiceChart = new Chart($("practice-chart"), {
    data: {
      datasets: [
        { type: "scatter", label: "One game", data: games, pointRadius: 2.5, pointHoverRadius: 5,
          borderWidth: 0, order: 2 },
        { type: "line", label: "Average", data: averages, borderWidth: 2, pointRadius: 0,
          pointHoverRadius: 4, tension: 0, order: 1 },
      ],
    },
    options: practiceOptions,
    plugins: [crosshair],
  });
  applyTheme();
}

function applyTheme() {
  for (const chart of [skillChart, practiceChart]) {
    if (!chart) continue;
    for (const axis of Object.values(chart.options.scales)) {
      axis.grid = { color: css("--grid") };
      axis.border = { color: css("--axis") };
      axis.ticks.color = css("--text-muted");
      axis.title.color = css("--text-secondary");
    }
    const tip = chart.options.plugins.tooltip;
    Object.assign(tip, { backgroundColor: css("--surface"), borderColor: css("--border"), borderWidth: 1,
                         titleColor: css("--text-secondary"), bodyColor: css("--text-primary") });
  }
  if (skillChart) {
    const [line, ref] = skillChart.data.datasets;
    Object.assign(line, { borderColor: css("--series"), pointBackgroundColor: css("--series"),
                          pointBorderColor: css("--surface"), pointHoverBackgroundColor: css("--series") });
    ref.borderColor = css("--reference");
    skillChart.update("none");
  }
  if (practiceChart) {
    const [dots, line] = practiceChart.data.datasets;
    dots.backgroundColor = css("--series-dots");
    Object.assign(line, { borderColor: css("--series"), pointHoverBackgroundColor: css("--series") });
    practiceChart.update("none");
  }
}

// ---------------------------------------------------------------- training data

function clearRun() {
  games.length = 0;
  averages.length = 0;
  skillPoints.length = 0;
  windowSum = 0;
  analysisKey = null;
  $("table-body").replaceChildren();
  $("analysis").hidden = true;
  $("lookup-result").replaceChildren();
}

function addGame([n, secret, guesses]) {
  games.push({ x: n, y: guesses, secret });
  windowSum += guesses;
  if (games.length > windowSize) windowSum -= games[games.length - windowSize - 1].y;
  const avg = windowSum / Math.min(games.length, windowSize);
  averages.push({ x: n, y: avg });

  const row = el("tr");
  for (const text of [n, secret.toUpperCase(), guesses, avg.toFixed(2)]) row.appendChild(el("td", "", text));
  $("table-body").prepend(row);
}

function renderKnowledge(knowledge) {
  if (knowledge.kind === "value") {
    $("knowledge-subtitle").textContent = "How many more guesses it expects to need, depending on how many words are " +
      "still possible. It learns this curve from its own games (3 numbers). A blank AI thinks it's 0 everywhere.";
    const top = Math.max(1, ...knowledge.items.map(([, v]) => v));
    const rows = knowledge.items.map(([m, v]) => {
      const row = el("div", "bar-row");
      row.appendChild(el("span", "", `${m.toLocaleString("en-US")} word${m === 1 ? "" : "s"} left`));
      const track = el("div", "track");
      const fill = el("div", "fill");
      fill.style.width = `${(100 * Math.max(v, 0)) / top}%`;
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", "num", `${v.toFixed(2)} more`));
      return row;
    });
    const box = el("div", "bars");
    box.replaceChildren(...rows);
    $("weights").replaceChildren(box);
    return;
  }
  $("knowledge-subtitle").textContent = 'Its whole "brain" is these numbers. All zero means a blank AI that picks at ' +
    "random. Positive means it wants more of that; negative means it wants less.";
  const weights = Object.fromEntries(knowledge.items);
  const values = Object.values(weights);
  const scale = Math.max(1, ...values.map(Math.abs));
  const rows = Object.entries(weights).map(([name, value]) => {
    const row = el("div", "weight");
    row.appendChild(el("span", "", WEIGHT_LABELS[name] || name));
    const track = el("div", "track");
    if (value !== 0) {
      const fill = el("div", `fill ${value > 0 ? "pos" : "neg"}`);
      fill.style.width = `${(Math.abs(value) / scale) * 50}%`;
      track.appendChild(fill);
    }
    row.appendChild(track);
    row.appendChild(el("span", "num", (value >= 0 ? "+" : "") + value.toFixed(2)));
    return row;
  });
  $("weights").replaceChildren(...rows);
}

function syncSlider(id, list, value, format) {
  if (dragging[id]) return;
  let index = list.indexOf(value);
  if (index < 0) index = list.reduce((best, v, i) => Math.abs(v - value) < Math.abs(list[best] - value) ? i : best, 0);
  $(id).value = index;
  $(`${id}-label`).textContent = format(value);
}
const formatSpeed = (speed) => speed === 0 ? "max speed" : `${speed} games/s`;
const formatTarget = (target) => `${target.toLocaleString("en-US")} games`;

function render(state) {
  if (!skillChart && !practiceChart && state.target_games) makeCharts(state.target_games);
  windowSize = state.window;

  if (state.run_id !== runId) { // first load, or someone pressed Reset
    runId = state.run_id;
    clearRun();
  }
  if (state.since === games.length) state.results.forEach(addGame); // skip replies that don't line up

  skillPoints.length = 0;
  for (const [n, score] of state.skill_checks) skillPoints.push({ x: n, y: score });

  for (const chart of [skillChart, practiceChart]) {
    if (chart) chart.options.scales.x.max = state.target_games;
  }
  if (skillChart) { // faint line at the blank AI's level, so the drop is easy to see
    const start = state.stats.skill_before;
    skillChart.data.datasets[1].data = start === null ? []
      : [{ x: 0, y: start }, { x: state.target_games, y: start }];
  }

  // Status + controls
  const labels = { running: "Running", paused: "Paused", finished: "Finished · model saved" };
  $("status").textContent = labels[state.status] || state.status;
  $("status").dataset.status = state.status;
  $("pause").disabled = state.status !== "running";
  $("continue").disabled = state.status !== "paused";
  syncSlider("speed", SPEEDS, state.games_per_second, formatSpeed);
  syncSlider("target", TARGETS, state.target_games, formatTarget);
  if (document.activeElement !== $("mode")) $("mode").value = state.mode;
  document.querySelectorAll("[data-mode]").forEach((node) => { node.hidden = node.dataset.mode !== state.mode; });

  // Tiles
  const s = state.stats;
  $("games").textContent = state.games_played.toLocaleString("en-US");
  $("games-sub").textContent = `of ${state.target_games.toLocaleString("en-US")}`;
  $("skill-before").textContent = s.skill_before === null ? "–" : s.skill_before.toFixed(2);
  $("skill-now").textContent = s.skill_now === null ? "–" : s.skill_now.toFixed(2);
  $("skill-now-sub").textContent = s.skill_now === null ? "" : `avg guesses after ${s.skill_now_games.toLocaleString("en-US")} games`;
  if (s.skill_now !== null && s.skill_before !== null) {
    const saved = s.skill_before - s.skill_now;
    $("improvement").textContent = `${saved >= 0 ? "−" : "+"}${Math.abs(saved).toFixed(2)}`;
    $("improvement-sub").textContent = `guesses per game (${((saved / s.skill_before) * 100).toFixed(1)}% fewer)`;
  }
  $("check-words").textContent = state.check_words;
  $("skill-subtitle").textContent = `Every 10 games at first (then every ${state.check_interval}), training pauses and the AI plays the same ${state.check_words} words. Same words every time, so the luck of which word comes up can't move these lines: only learning can. Game 0 is the blank AI, before any training.`;
  $("avg-legend").textContent = `Average of the last ${windowSize} games`;
  $("table-avg-head").textContent = `Avg of last ${windowSize}`;
  renderKnowledge(state.knowledge);
  renderWords(state);
  const optimum = state.words.optimum;  // proven best TOTAL over every answer (official list only)
  const count = (n) => n.toLocaleString("en-US");
  const vsOptimum = (avg) => {
    if (!optimum) return "no published optimum for this list";
    const total = Math.round(avg * state.words.answers);
    return total === optimum
      ? `${count(total)} guesses in total: perfect play (the proven optimum)`
      : `${count(total)} guesses in total; perfect play is ${count(optimum)} (${total > optimum ? "+" : ""}${count(total - optimum)})`;
  };
  const exam = state.final_exam;
  $("exam").textContent = exam ? exam.avg.toFixed(3) : "–";
  $("exam-sub").textContent = exam
    ? `worst game ${exam.worst}; ${vsOptimum(exam.avg)}` +
      (state.mode === "possible" && optimum ? ", but that needs probe words, which this mode can't play" : "")
    : "when the run finishes";
  const la = state.lookahead;
  $("run-lookahead").disabled = state.mode !== "any" || la.state === "running" || state.games_played === 0;
  $("lookahead").textContent = la.state === "done" ? la.avg.toFixed(3) : la.state === "running" ? "…" : "–";
  $("lookahead-sub").textContent = state.mode !== "any" ? "needs \"Any valid word\" mode"
    : la.state === "running" ? `thinking ahead on every word (the AI after ${la.games_trained} games)…`
    : la.state === "done" ? `after ${la.games_trained} games, opener ${la.opener.toUpperCase()}, ` +
      `${Math.round(la.seconds)} s; ${vsOptimum(la.avg)}`
    : "same AI, weighing its top 10 guesses exactly each turn";
  const se = state.search;
  $("run-search").disabled = state.mode !== "any" || se.state === "running" || state.games_played === 0;
  $("search").textContent = se.state === "done" ? se.avg.toFixed(3) : se.state === "running" ? "…" : "–";
  $("search-sub").textContent = state.mode !== "any" ? "needs \"Any valid word\" mode"
    : se.state === "running"
      ? (se.openers ? `opener ${se.done} of ${se.openers} worked out` + (se.best ? `; best so far ${count(se.best)} guesses in total` : "")
        : "ranking openers…")
    : se.state === "done" ? `after ${se.games_trained} games, opener ${se.opener.toUpperCase()}, ` +
      `${Math.round(se.seconds)} s; ${vsOptimum(se.avg)}`
    : se.state === "error" ? "the search stopped unexpectedly"
    : "same AI's top 20 guesses at every position, worked out exactly to the end of every game";
  renderLearned(state.learned, state.mode);
  renderOpeners(state.openers);
  renderAnalysisStatus(state.analysis);

  if (skillChart) skillChart.update("none");
  if (practiceChart) practiceChart.update("none");
}

// ---------------------------------------------------------------- word length

let nAnswers = 2315;

function renderWords(state) {
  const w = state.words;
  nAnswers = w.answers;
  document.querySelectorAll(".n-answers").forEach((node) => { node.textContent = w.answers.toLocaleString("en-US"); });
  document.querySelectorAll(".n-guesses").forEach((node) => { node.textContent = w.guesses.toLocaleString("en-US"); });
  if (!dragging.length) {
    const shown = w.preparing ? Number(w.preparing.replace(/\D/g, "")) : w.length;
    $("length").value = shown;
    $("length-label").textContent = `${shown} letters`;
    $("official-box").hidden = shown !== 5;
    if (!w.preparing) $("official").checked = w.official;
  }
  $("words-info").textContent = w.error ? `Couldn't prepare that list: ${w.error}`
    : w.preparing ? `Preparing ${w.preparing.replace(/\D/g, "")}-letter words… (the first time can take ~20 s)`
    : `${w.official ? "" : "Common words: "}${w.answers.toLocaleString("en-US")} possible answers · ${w.guesses.toLocaleString("en-US")} allowed guesses`;
  if (w.preparing) {
    $("status").textContent = "Preparing words…";
    $("status").dataset.status = "paused";
  }
  $("lookup-word").maxLength = w.length;
  $("lookup-word").placeholder = `e.g. ${EXAMPLES[w.length]}`;
  if (wordsName !== null && wordsName !== w.name) $("lookup-result").replaceChildren();
  wordsName = w.name;
}

function postWords() {
  const length = Number($("length").value);
  post("/api/words", { length, official: length === 5 && $("official").checked });
}

// ---------------------------------------------------------------- what it learned to do

const pct = (part, whole) => whole ? `${Math.round((100 * part) / whole)}%` : "–";
const upper = (words) => words.map((w) => w.toUpperCase()).join(", ");

function renderLearned(learned, mode) {
  $("learned-subtitle").textContent = `From its last ${learned.window.toLocaleString("en-US")} practice games.` +
    (mode === "possible" ? " In this mode it isn't allowed to probe." : "");
  const rows = learned.probing.map((b) => {
    const row = el("div", "bar-row");
    row.appendChild(el("span", "", b.label));
    const track = el("div", "track");
    const fill = el("div", "fill");
    fill.style.width = b.decisions ? `${(100 * b.probes) / b.decisions}%` : "0%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", "num", b.decisions ? `${pct(b.probes, b.decisions)} of ${b.decisions}` : "no guesses yet"));
    return row;
  });
  $("probing").replaceChildren(...rows);

  const p = learned.latest_probe;
  if (p) {
    const box = el("div");
    box.appendChild(document.createTextNode(`Game ${p.game.toLocaleString("en-US")}: ${p.possible.length} words were possible (`));
    box.appendChild(el("span", "probe-words", upper(p.possible)));
    box.appendChild(document.createTextNode("). It played "));
    box.appendChild(el("b", "", p.guess.toUpperCase()));
    box.appendChild(document.createTextNode(
      `, which can't be the answer, to test ${p.tested.length ? upper(p.tested) : "letters"} at once. ` +
      `Afterwards ${p.left_after} word${p.left_after === 1 ? " was" : "s were"} left (the secret was ${p.secret.toUpperCase()}).`));
    $("latest-probe").replaceChildren(box);
  } else {
    $("latest-probe").textContent = mode === "possible" ? "Probes aren't allowed in this mode." : "No probe yet.";
  }

  $("weak-opener").textContent = learned.weak_openers
    ? `When its opener still left over ${learned.weak_threshold} words (${learned.weak_openers} games), its next guess ` +
      `used only brand-new letters ${pct(learned.weak_fresh, learned.weak_openers)} of the time.`
    : "No weak openers yet.";
}

function renderOpeners(openers) {
  $("openers-subtitle").textContent =
    `${openers.tried.toLocaleString("en-US")} different openers so far. Its favorite right now: ${openers.favorite.toUpperCase()}.`;
  const rows = openers.table.map((o) => {
    const row = el("tr");
    row.appendChild(el("td", "", o.word.toUpperCase()));
    row.appendChild(el("td", "", o.games.toLocaleString("en-US")));
    row.appendChild(el("td", "", o.avg.toFixed(2) + (o.ci === null ? "" : ` \u00b1 ${o.ci.toFixed(2)}`)));
    return row;
  });
  $("openers-body").replaceChildren(...rows);
  $("openers-note").textContent = openers.games_to_compare
    ? "Single games vary a lot, so telling apart two openers that are 0.05 guesses apart would take about " +
      `${openers.games_to_compare.toLocaleString("en-US")} games with each. That's why the benchmark scores openers ` +
      `exactly, on all ${nAnswers.toLocaleString("en-US")} words, instead of by trial and error.`
    : "";
}

// ---------------------------------------------------------------- word difficulty

function renderAnalysisStatus(analysis) {
  const running = analysis.state === "running";
  $("analyze").disabled = running;
  $("analysis-status").textContent = running ? `Analyzing… ${Math.round(analysis.progress * 100)}%` : "";
  if (analysis.state === "done") {
    const key = `${runId}:${analysis.games_trained}`;
    if (key !== analysisKey) {
      analysisKey = key;
      fetch("/api/analysis").then((r) => r.json()).then((a) => a.result && renderAnalysis(a));
    }
  } else if (analysis.state === "idle") {
    $("analysis").hidden = true;
  }
}

function renderAnalysis({ result, games_trained: trained }) {
  const top = result.conclusions[0];
  $("analysis-summary").textContent =
    `All ${result.words.toLocaleString("en-US")} words, played by the AI after ${trained.toLocaleString("en-US")} games of training: ` +
    `${result.avg_guesses.toFixed(2)} guesses on average, and its opening word is ${result.opener.toUpperCase()}. ` +
    `Biggest factor: ${top.title}. ${top.hard_label}: ${top.hard.toFixed(2)} guesses vs ` +
    `${top.easy_label}: ${top.easy.toFixed(2)}.`;

  // One dot plot per trait, all on the same scale so they compare fairly.
  const all = Object.values(result.groups).flat().filter((g) => g.avg_guesses !== null).map((g) => g.avg_guesses);
  const lo = Math.floor((Math.min(...all) - 0.1) * 10) / 10;
  const hi = Math.ceil((Math.max(...all) + 0.1) * 10) / 10;
  const blocks = result.conclusions.map((c, i) => {
    const block = el("div", "trait");
    const title = el("h4", "", `${c.title} `);
    title.appendChild(el("span", "rank", `· ${c.gap.toFixed(2)} guess gap${i === 0 ? " (biggest)" : ""}`));
    block.appendChild(title);
    for (const g of result.groups[c.trait]) {
      if (g.avg_guesses === null) continue;
      const row = el("div", "dotrow");
      row.title = `${g.words.toLocaleString("en-US")} words`;
      row.appendChild(el("span", "", `${g.label} (${g.words.toLocaleString("en-US")})`));
      const track = el("div", "track");
      const dot = el("div", "dot");
      dot.style.left = `${((g.avg_guesses - lo) / (hi - lo)) * 100}%`;
      track.appendChild(dot);
      row.appendChild(track);
      row.appendChild(el("span", "num", g.avg_guesses.toFixed(2)));
      block.appendChild(row);
    }
    const scale = el("div", "scale");
    scale.appendChild(el("span", "", "avg guesses"));
    const ends = el("div");
    ends.appendChild(el("span", "", lo.toFixed(1)));
    ends.appendChild(el("span", "", hi.toFixed(1)));
    scale.appendChild(ends);
    scale.appendChild(el("span"));
    block.appendChild(scale);
    return block;
  });
  $("traits").replaceChildren(...blocks);

  const rows = result.hardest.map((h) => {
    const row = el("tr");
    const wordCell = el("td");
    const link = el("a", "", h.word.toUpperCase());
    link.addEventListener("click", () => lookup(h.word));
    wordCell.appendChild(link);
    row.appendChild(wordCell);
    row.appendChild(el("td", "", h.guesses));
    row.appendChild(el("td", "", h.left_after_opener.toLocaleString("en-US")));
    row.appendChild(el("td", "", h.repeated_letters ? "yes" : "no"));
    const alikes = h.look_alikes.slice(0, 6).map((w) => w.toUpperCase()).join(", ");
    row.appendChild(el("td", "", h.look_alikes.length ? alikes + (h.look_alikes.length > 6 ? ` +${h.look_alikes.length - 6}` : "") : "none"));
    return row;
  });
  $("hardest-body").replaceChildren(...rows);
  $("analysis").hidden = false;
}

async function lookup(word) {
  $("lookup-word").value = word;
  const report = await (await fetch(`/api/word?w=${encodeURIComponent(word)}`)).json();
  const box = $("lookup-result");
  if (report.error) {
    box.replaceChildren(el("p", "explain", report.error));
    return;
  }
  const n = report.steps.length;
  const head = el("p", "explain",
    `${report.word.toUpperCase()}: solved in ${n} guess${n === 1 ? "" : "es"} by the AI after ` +
    `${report.games_trained.toLocaleString("en-US")} games of training. Repeated letter: ${report.repeated_letters ? "yes" : "no"}. ` +
    `One-letter look-alikes: ${report.look_alikes.length ? report.look_alikes.map((w) => w.toUpperCase()).join(", ") : "none"}.`);
  const rows = report.steps.map((step, i) => {
    const row = el("div", "play-row");
    const tiles = el("div", "tiles-row");
    [...step.guess].forEach((letter, j) => tiles.appendChild(el("div", `wtile g${step.feedback[j]}`, letter.toUpperCase())));
    row.appendChild(tiles);
    row.appendChild(el("span", "", i === n - 1 ? "solved"
      : `${step.words_left.toLocaleString("en-US")} word${step.words_left === 1 ? "" : "s"} still possible`));
    if (step.probe) row.appendChild(el("span", "badge", "probe: couldn't be the answer"));
    return row;
  });
  box.replaceChildren(head, ...rows);
}

// ---------------------------------------------------------------- server

async function poll() {
  try {
    const response = await fetch(`/api/state?since=${games.length}`);
    render(await response.json());
  } catch (error) {
    $("status").textContent = "Disconnected (is dashboard.py still running?)";
    $("status").dataset.status = "disconnected";
  }
  setTimeout(poll, POLL_MS);
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  render(await response.json());
}

function slider(id, list, format, path, key) {
  $(id).addEventListener("input", () => {
    dragging[id] = true;
    $(`${id}-label`).textContent = format(list[$(id).value]);
  });
  $(id).addEventListener("change", async () => {
    await post(path, { [key]: list[$(id).value] });
    dragging[id] = false;
  });
}

$("pause").addEventListener("click", () => post("/api/pause"));
$("continue").addEventListener("click", () => post("/api/continue"));
$("reset").addEventListener("click", () => post("/api/reset"));
$("analyze").addEventListener("click", () => post("/api/analyze"));
$("mode").addEventListener("change", () => post("/api/mode", { mode: $("mode").value }));
$("run-lookahead").addEventListener("click", () => post("/api/lookahead"));
$("run-search").addEventListener("click", () => post("/api/search"));
$("length").addEventListener("input", () => {
  dragging.length = true;
  $("length-label").textContent = `${$("length").value} letters`;
  $("official-box").hidden = Number($("length").value) !== 5;
});
$("length").addEventListener("change", async () => { await postWords(); dragging.length = false; });
$("official").addEventListener("change", postWords);
$("lookup").addEventListener("submit", (event) => {
  event.preventDefault();
  const word = $("lookup-word").value.trim();
  if (word) lookup(word);
});
slider("speed", SPEEDS, formatSpeed, "/api/speed", "games_per_second");
slider("target", TARGETS, formatTarget, "/api/target", "games");
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

const linkedWord = new URLSearchParams(location.search).get("word"); // e.g. /?word=foyer
if (linkedWord) lookup(linkedWord);
poll();
