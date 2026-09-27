// Live training dashboard: polls the Python server and draws one dot per game
// plus a rolling average. All the learning happens in Python (live/session.py).

const SPEEDS = [1, 2, 5, 10, 25, 50, 100, 0]; // games per second; 0 = as fast as possible
const POLL_MS = 250;

let runId = null;
let windowSize = 50;
let games = []; // {x: game number, y: guesses, secret}
let averages = []; // {x: game number, y: average of the last `windowSize` games}
let windowSum = 0;
let draggingSpeed = false;

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- chart

const crosshair = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const active = chart.tooltip && chart.tooltip.getActiveElements();
    if (!active || !active.length) return;
    const x = active[0].element.x;
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = css("--axis");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function makeChart(targetGames) {
  if (!window.Chart) {
    $("chart-error").hidden = false;
    return null;
  }
  return new Chart($("chart"), {
    data: {
      datasets: [
        { type: "scatter", label: "One game", data: games, pointRadius: 3, pointHoverRadius: 5,
          borderWidth: 0, order: 2 },
        { type: "line", label: "Average", data: averages, borderWidth: 2, pointRadius: 0,
          pointHoverRadius: 4, tension: 0, borderJoinStyle: "round", borderCapStyle: "round", order: 1 },
        { type: "line", label: "No strategy", data: [], borderWidth: 1, pointRadius: 0,
          pointHoverRadius: 0, order: 3 },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "linear", min: 0, max: targetGames, ticks: {},
             title: { display: true, text: "games played" } },
        y: { min: 1, suggestedMax: 8, ticks: { stepSize: 1, precision: 0 },
             title: { display: true, text: "guesses to find the word" } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => item.datasetIndex !== 2,
          displayColors: false,
          callbacks: {
            title: (items) => `Game ${items[0].raw.x}`,
            label: (item) => item.datasetIndex === 0
              ? `${item.raw.y} guesses · ${item.raw.secret.toUpperCase()}`
              : `${item.raw.y.toFixed(2)} average of the last ${Math.min(windowSize, item.raw.x)}`,
          },
        },
      },
    },
    plugins: [crosshair],
  });
}

function applyTheme(chart) {
  if (!chart) return;
  const [dots, line, ref] = chart.data.datasets;
  dots.backgroundColor = css("--series-dots");
  line.borderColor = css("--series");
  line.pointHoverBackgroundColor = css("--series");
  ref.borderColor = css("--reference");
  for (const axis of Object.values(chart.options.scales)) {
    axis.grid = { color: css("--grid") };
    axis.border = { color: css("--axis") };
    axis.ticks.color = css("--text-muted");
    axis.title.color = css("--text-secondary");
  }
  const tip = chart.options.plugins.tooltip;
  tip.backgroundColor = css("--surface");
  tip.borderColor = css("--border");
  tip.borderWidth = 1;
  tip.titleColor = css("--text-secondary");
  tip.bodyColor = css("--text-primary");
  chart.update("none");
}

let chart = null;

// ---------------------------------------------------------------- data

function clearRun() {
  games.length = 0;
  averages.length = 0;
  windowSum = 0;
  $("table-body").replaceChildren();
}

function addGame([n, secret, guesses]) {
  games.push({ x: n, y: guesses, secret });
  windowSum += guesses;
  if (games.length > windowSize) windowSum -= games[games.length - windowSize - 1].y;
  const avg = windowSum / Math.min(games.length, windowSize);
  averages.push({ x: n, y: avg });

  const row = document.createElement("tr");
  for (const text of [n, secret.toUpperCase(), guesses, avg.toFixed(2)]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.appendChild(cell);
  }
  $("table-body").prepend(row);
}

function render(state) {
  if (!chart && state.target_games) {
    chart = makeChart(state.target_games);
    applyTheme(chart);
  }
  windowSize = state.window;

  if (state.run_id !== runId) { // first load, or someone pressed Reset
    runId = state.run_id;
    clearRun();
  }
  if (state.since === games.length) { // ignore replies that don't line up with what we have
    state.results.forEach(addGame);
  }

  // Status + buttons
  const labels = { running: "Running", paused: "Paused", finished: "Finished · model saved" };
  $("status").textContent = labels[state.status] || state.status;
  $("status").dataset.status = state.status;
  $("pause").disabled = state.status !== "running";
  $("continue").disabled = state.status !== "paused";

  if (!draggingSpeed) {
    const index = SPEEDS.indexOf(state.games_per_second);
    $("speed").value = index >= 0 ? index : SPEEDS.indexOf(10);
    showSpeed();
  }

  // Tiles
  const s = state.stats;
  $("games").textContent = state.games_played.toLocaleString();
  $("games-sub").textContent = `of ${state.target_games.toLocaleString()}`;
  $("last").textContent = s.last_game ? `${s.last_game.guesses} guesses` : "–";
  $("last-sub").textContent = s.last_game ? s.last_game.secret.toUpperCase() : "";
  $("first").textContent = s.first_avg === null ? "–" : s.first_avg.toFixed(2);
  $("recent").textContent = s.recent_avg === null ? "–" : s.recent_avg.toFixed(2);
  $("first-label").textContent = `First ${s.first_count || windowSize} games`;
  $("recent-label").textContent = `Last ${s.recent_count || windowSize} games`;
  $("avg-legend").textContent = `Average of the last ${windowSize} games`;
  $("table-avg-head").textContent = `Avg of last ${windowSize}`;

  // Reference line: how many guesses it takes with no strategy at all
  if (chart && state.baseline !== null && chart.data.datasets[2].data.length === 0) {
    chart.data.datasets[2].data = [{ x: 0, y: state.baseline }, { x: state.target_games, y: state.baseline }];
    $("ref-legend").textContent = `No strategy (random possible word): ${state.baseline.toFixed(2)}`;
  }
  if (chart) chart.update("none");
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

function showSpeed() {
  const speed = SPEEDS[$("speed").value];
  $("speed-label").textContent = speed === 0 ? "max speed" : `${speed} games/s`;
}

$("pause").addEventListener("click", () => post("/api/pause"));
$("continue").addEventListener("click", () => post("/api/continue"));
$("reset").addEventListener("click", () => post("/api/reset"));
$("speed").addEventListener("input", () => { draggingSpeed = true; showSpeed(); });
$("speed").addEventListener("change", async () => {
  await post("/api/speed", { games_per_second: SPEEDS[$("speed").value] });
  draggingSpeed = false;
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(chart));

poll();
