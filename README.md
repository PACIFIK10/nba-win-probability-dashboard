# NBA Live Win Probability Dashboard

A full-stack ML project: predicts a live win probability for the home team
as an NBA game unfolds, and streams it to a real-time dashboard.

**Tech stack:** `nba_api` (play-by-play logs) → `pandas` (feature pipeline)
→ `PyTorch` (neural network) → `Flask` (serving) → `WebSocket` (real-time
push to the dashboard).

<img width="794" height="853" alt="image" src="https://github.com/user-attachments/assets/f0e2ce1c-b173-481a-b009-8f10c4276e6d" />

## How it works

1. **Data** — `src/fetch_data.py` pulls play-by-play logs + final scores
   for a batch of games via `nba_api`.
2. **Features** — `src/features.py` turns each event into a snapshot of
   4 signals: score differential, seconds remaining, possession, and team
   fouls (+ bonus flag), labeled with whether the home team ultimately won.
3. **Model** — `src/model.py` / `src/train.py` train a small feedforward
   neural network (PyTorch) to map a game snapshot → home win probability.
4. **Serving** — `src/serve.py` is a Flask app that loads the trained
   model and exposes:
   - `GET /api/games` — list of games available to replay
   - `GET /api/predict/<score_diff>/<seconds_remaining>/<possession_home>/<home_fouls>/<away_fouls>` — one-off prediction
   - `WS /ws/live/<game_id>` — streams the game event-by-event with a live
     win probability attached to each event
5. **Dashboard** — `templates/dashboard.html` + `static/dashboard.js`
   connect to the WebSocket and render a live-updating probability chart,
   scoreboard, foul/bonus tracker, and play-by-play log (Chart.js).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 1) Get data. Two options:
python src/generate_synthetic_data.py --n-games 80   # quick local demo data
# -- or, for real games --
python src/fetch_data.py --season 2023-24 --n-games 25

# 2) Build features
python src/features.py

# 3) Train the model
python src/train.py --epochs 30

# 4) Run the dashboard
python src/serve.py
# open http://localhost:5000
```

Pick a game from the dropdown and hit **Start Replay** — it streams that
game's play-by-play in near real time, with the win probability line
updating on every event.

## Deploying (so it runs without anyone cloning it)

This repo is ready to deploy to [Render](https://render.com) (free tier,
no credit card) via `render.yaml`:

1. Push this repo to GitHub.
2. On Render: **New** → **Blueprint** → connect your GitHub repo → Render
   reads `render.yaml` and sets everything up automatically.
3. First deploy takes a few minutes (installing torch + building). After
   that you get a public URL like `https://nba-win-probability-dashboard.onrender.com`.

Notes:
- `requirements.txt` installs the **CPU-only** build of PyTorch
  (`--extra-index-url https://download.pytorch.org/whl/cpu`) — the app only
  runs inference in production, not training, so this keeps the deploy
  small and fast.
- Serving uses `gunicorn` with the `gevent` worker (`Procfile` /
  `render.yaml` startCommand) instead of Flask's dev server, since
  `flask-sock`'s WebSocket route needs a worker that supports it.
- The free Render tier spins down after inactivity and takes ~30s to wake
  on the next visit — normal for a free-tier demo link.
- The data and trained model are committed to the repo, so the deployed
  app works immediately with no setup step.

## Project structure

```
nba-win-probability-dashboard/
├── data/
│   ├── raw/            # play_by_play.csv, game_labels.csv
│   └── processed/      # features.csv
├── models/              # win_probability_net.pt, norm_stats.json
├── src/
│   ├── fetch_data.py            # real nba_api data pull
│   ├── generate_synthetic_data.py  # offline stand-in, same schema
│   ├── features.py              # event -> feature-row engineering
│   ├── model.py                 # PyTorch WinProbabilityNet
│   ├── train.py                 # training loop + eval
│   ├── inference.py             # load model, predict(features) -> prob
│   └── serve.py                 # Flask app + WebSocket replay/live stream
├── templates/dashboard.html
├── static/{style.css,dashboard.js}
└── requirements.txt
```
