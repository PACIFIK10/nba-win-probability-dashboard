"""
serve.py
--------
The "SERVER" (flask, serve predictions) + "LIVE" (websocket, real-time
dashboard) layers of the tech stack.

Two ways to feed it live games:

1. REPLAY MODE (default, works fully offline): picks a game out of
   data/raw/play_by_play.csv and replays its events over a WebSocket,
   computing win probability at each event, paced with a short delay so
   it feels like a live broadcast. This is what the bundled dashboard
   uses out of the box.

2. LIVE MODE (needs real internet access to nba.com's live endpoints):
   see `stream_from_nba_live()` below for the hook point -- swap the event
   source from the CSV replay to `nba_api.live.nba.endpoints.playbyplay`
   polling and everything downstream (feature building + inference +
   websocket push) is unchanged.

Run:
    python src/serve.py
Then open http://localhost:5000
"""
import json
import os
import time

from flask import Flask, jsonify, render_template
from flask_sock import Sock

import pandas as pd

from features import build_features_for_game
from inference import get_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
sock = Sock(app)

_pbp_cache = None
_labels_cache = None


def load_data():
    global _pbp_cache, _labels_cache
    if _pbp_cache is None:
        pbp_path = os.path.join(RAW_DIR, "play_by_play.csv")
        labels_path = os.path.join(RAW_DIR, "game_labels.csv")
        if not os.path.exists(pbp_path):
            raise FileNotFoundError(
                "No play-by-play data found. Run `python src/generate_synthetic_data.py` "
                "(offline demo) or `python src/fetch_data.py` (real nba_api data) first."
            )
        _pbp_cache = pd.read_csv(pbp_path)
        _labels_cache = pd.read_csv(labels_path).set_index("GAME_ID")
    return _pbp_cache, _labels_cache


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/games")
def api_games():
    pbp, labels = load_data()
    games = []
    for game_id, row in labels.iterrows():
        games.append({
            "game_id": game_id,
            "home": str(row["HOME_TEAM_ID"]),
            "away": str(row["AWAY_TEAM_ID"]),
            "home_pts": int(row["HOME_PTS"]),
            "away_pts": int(row["AWAY_PTS"]),
        })
    return jsonify(games)


@app.route("/api/predict/<path:score_diff>/<seconds_remaining>/<possession_home>/<home_fouls>/<away_fouls>")
def api_predict(score_diff, seconds_remaining, possession_home, home_fouls, away_fouls):
    """Simple GET-based prediction endpoint for manual testing / curl."""
    model = get_model()
    home_fouls = int(home_fouls)
    away_fouls = int(away_fouls)
    features = {
        "score_diff": float(score_diff),
        "seconds_remaining": float(seconds_remaining),
        "possession_home": int(possession_home),
        "home_fouls": home_fouls,
        "away_fouls": away_fouls,
        "home_bonus": int(home_fouls >= 5),
        "away_bonus": int(away_fouls >= 5),
    }
    prob = model.predict(features)
    return jsonify({"home_win_probability": prob, "features": features})


@sock.route("/ws/live/<game_id>")
def ws_live(ws, game_id):
    """
    Replays one game's play-by-play in real time, pushing a JSON message
    per event: {period, clock, score_diff, home_fouls, away_fouls,
    win_prob, event_desc}.
    """
    pbp, labels = load_data()
    model = get_model()

    game_df = pbp[pbp["GAME_ID"].astype(str) == str(game_id)]
    if game_df.empty:
        # try numeric match (synthetic IDs are strings like SYN00001)
        game_df = pbp[pbp["GAME_ID"] == game_id]
    if game_df.empty:
        ws.send(json.dumps({"error": f"game {game_id} not found"}))
        return

    label_row = labels.loc[game_id] if game_id in labels.index else None
    home_win = int(label_row["HOME_WIN"]) if label_row is not None else None

    feats = build_features_for_game(game_df, home_win or 0)
    game_df = game_df.sort_values("EVENTNUM").reset_index(drop=True)

    speed = 0.05  # seconds between events; tune for demo pacing

    for i, feat_row in feats.iterrows():
        ev = game_df.iloc[i]
        feature_dict = {col: feat_row[col] for col in
                         ["score_diff", "seconds_remaining", "possession_home",
                          "home_fouls", "away_fouls", "home_bonus", "away_bonus"]}
        prob = model.predict(feature_dict)

        desc = ev.get("HOMEDESCRIPTION")
        if not isinstance(desc, str):
            desc = ev.get("VISITORDESCRIPTION")
        if not isinstance(desc, str):
            desc = ev.get("NEUTRALDESCRIPTION") or ""

        payload = {
            "period": int(ev["PERIOD"]),
            "clock": ev.get("PCTIMESTRING", ""),
            "score_diff": int(feature_dict["score_diff"]),
            "home_fouls": int(feature_dict["home_fouls"]),
            "away_fouls": int(feature_dict["away_fouls"]),
            "possession_home": int(feature_dict["possession_home"]),
            "home_win_probability": round(prob, 4),
            "event": desc,
            "final": bool(i == len(feats) - 1),
        }
        try:
            ws.send(json.dumps(payload))
        except Exception:
            break
        time.sleep(speed)


def stream_from_nba_live(game_id: str):
    """
    Hook point for a REAL live feed. Not called by default (requires
    network access to nba.com's live-data endpoints, which most sandboxed
    environments can't reach). On a normal machine you'd do:

        from nba_api.live.nba.endpoints import playbyplay
        import time
        seen = set()
        while True:
            pbp = playbyplay.PlayByPlay(game_id).get_dict()
            for action in pbp["game"]["actions"]:
                if action["actionNumber"] not in seen:
                    seen.add(action["actionNumber"])
                    yield action
            time.sleep(3)

    ...then map each `action` into the same feature dict shape used above
    and push it over the websocket exactly like ws_live() does.
    """
    raise NotImplementedError("See docstring - wire this up on a machine with live internet access.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
