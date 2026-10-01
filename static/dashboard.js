const gameSelect = document.getElementById("game-select");
const startBtn = document.getElementById("start-btn");
const statusPill = document.getElementById("status-pill");
const homeLabel = document.getElementById("home-label");
const awayLabel = document.getElementById("away-label");
const homeProb = document.getElementById("home-prob");
const awayProb = document.getElementById("away-prob");
const clockEl = document.getElementById("clock");
const scoreDiffEl = document.getElementById("score-diff");
const possessionEl = document.getElementById("possession");
const homeFoulsEl = document.getElementById("home-fouls");
const awayFoulsEl = document.getElementById("away-fouls");
const homeBonusEl = document.getElementById("home-bonus");
const awayBonusEl = document.getElementById("away-bonus");
const eventLog = document.getElementById("event-log");

let chart;
let ws;
let games = [];

function initChart() {
  const ctx = document.getElementById("prob-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Home win probability",
        data: [],
        borderColor: "#f0563a",
        backgroundColor: "rgba(240,86,58,0.12)",
        fill: true,
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#8a92a6", callback: v => `${Math.round(v * 100)}%` }, grid: { color: "#232838" } },
        x: { ticks: { display: false }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

async function loadGames() {
  const res = await fetch("/api/games");
  games = await res.json();
  gameSelect.innerHTML = games.map(g =>
    `<option value="${g.game_id}">${g.away} @ ${g.home} (final ${g.away_pts}-${g.home_pts})</option>`
  ).join("");
}

function resetUI(game) {
  homeLabel.textContent = game.home;
  awayLabel.textContent = game.away;
  homeProb.textContent = "50%";
  awayProb.textContent = "50%";
  clockEl.textContent = "12:00 · Q1";
  scoreDiffEl.textContent = "0";
  possessionEl.textContent = "—";
  homeFoulsEl.textContent = "0";
  awayFoulsEl.textContent = "0";
  homeBonusEl.textContent = "";
  awayBonusEl.textContent = "";
  eventLog.innerHTML = "";
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.update();
}

function startReplay() {
  const gameId = gameSelect.value;
  const game = games.find(g => g.game_id === gameId);
  if (!game) return;

  resetUI(game);
  startBtn.disabled = true;
  statusPill.textContent = "live";
  statusPill.classList.add("live");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/live/${gameId}`);

  ws.onmessage = (msg) => {
    const d = JSON.parse(msg.data);
    if (d.error) {
      statusPill.textContent = "error";
      return;
    }

    const hp = d.home_win_probability;
    const ap = 1 - hp;
    homeProb.textContent = `${Math.round(hp * 100)}%`;
    awayProb.textContent = `${Math.round(ap * 100)}%`;
    clockEl.textContent = `${d.clock || "0:00"} · Q${d.period}`;
    scoreDiffEl.textContent = d.score_diff > 0 ? `+${d.score_diff} HOME` : d.score_diff < 0 ? `${d.score_diff} AWAY` : "TIED";
    possessionEl.textContent = d.possession_home ? `⬤ ${game.home} ball` : `⬤ ${game.away} ball`;
    homeFoulsEl.textContent = d.home_fouls;
    awayFoulsEl.textContent = d.away_fouls;
    homeBonusEl.textContent = d.home_fouls >= 5 ? "BONUS" : "";
    awayBonusEl.textContent = d.away_fouls >= 5 ? "BONUS" : "";

    chart.data.labels.push("");
    chart.data.datasets[0].data.push(hp);
    if (chart.data.labels.length > 400) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update("none");

    if (d.event) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="t">Q${d.period} ${d.clock}</span>${d.event}`;
      eventLog.prepend(li);
      while (eventLog.children.length > 40) eventLog.removeChild(eventLog.lastChild);
    }

    if (d.final) {
      statusPill.textContent = "final";
      statusPill.classList.remove("live");
      startBtn.disabled = false;
    }
  };

  ws.onclose = () => {
    startBtn.disabled = false;
    if (statusPill.textContent === "live") {
      statusPill.textContent = "disconnected";
      statusPill.classList.remove("live");
    }
  };
}

startBtn.addEventListener("click", startReplay);

(async function init() {
  initChart();
  await loadGames();
})();
