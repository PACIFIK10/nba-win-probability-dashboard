# NBA Live Win Probability Dashboard

Built a full-stack machine learning app that predicts a live win probability for NBA games as they happen. A PyTorch neural network, trained on real play-by-play data (score differential, time remaining, possession, and fouls) from the NBA's official API, serves predictions through a Flask backend. A WebSocket connection pushes updated probabilities to a React-style dashboard in real time, so the win percentage shifts with every shot, foul, and turnover, just like the live odds you'd see on a sports broadcast.

**Tech stack:** `nba_api` (play-by-play logs) → `pandas` (feature pipeline)
→ `PyTorch` (neural network) → `Flask` (serving) → `WebSocket` (real-time
push to the dashboard).

<img width="780" height="852" alt="image" src="https://github.com/user-attachments/assets/1e6da7cd-cbf3-4cf3-be1c-b90e1239ac4f" />

## How it works

1. **Data** — `src/fetch_data.py` pulls play-by-play logs and final scores
   for a batch of games via `nba_api`.
2. **Features** — `src/features.py` turns each event into a snapshot of
   4 signals: score differential, seconds remaining, possession, and team
   fouls (+ bonus flag), labeled with whether the home team ultimately won.
3. **Model** — `src/model.py` / `src/train.py` train a small feedforward
   neural network (PyTorch) that maps a game snapshot to a home win
   probability.
4. **Serving** — `src/serve.py` is a Flask app that loads the trained
   model and exposes:
   - `GET /api/games` — list of games available to replay
   - `GET /api/predict/<score_diff>/<seconds_remaining>/<possession_home>/<home_fouls>/<away_fouls>` — one-off prediction
   - `WS /ws/live/<game_id>` — streams a game event-by-event with a live
     win probability attached to each event
5. **Dashboard** — `templates/dashboard.html` + `static/dashboard.js`
   connect to the WebSocket and render a live-updating probability chart,
   scoreboard, foul/bonus tracker, and play-by-play log (Chart.js).

Deployed on [Render](https://render.com), served with `gunicorn` and the
`gevent` worker so the WebSocket route stays responsive under a real
production server rather than Flask's development server.

## Project structure

```
nba-win-probability-dashboard/
├── data/
│   ├── raw/            # play_by_play.csv, game_labels.csv
│   └── processed/      # features.csv
├── models/              # win_probability_net.pt, norm_stats.json
├── src/
│   ├── fetch_data.py            # nba_api data pull
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
