"""
generate_synthetic_data.py
---------------------------
Generates synthetic play-by-play data in the SAME schema that
nba_api.stats.endpoints.playbyplayv2.PlayByPlayV2 returns, so the rest of
the pipeline (feature engineering -> PyTorch model -> Flask/WebSocket
dashboard) can be built and exercised end-to-end without needing to reach
stats.nba.com.

This is a stand-in for `fetch_data.py` during development / in sandboxed
environments. Swap in real data by running `fetch_data.py` on a machine
with normal internet access -> the output CSVs (data/raw/play_by_play.csv,
data/raw/game_labels.csv) have identical columns, so nothing downstream
needs to change.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

EVENTMSGTYPE = {
    "MADE_SHOT": 1,
    "MISSED_SHOT": 2,
    "FREE_THROW": 3,
    "REBOUND": 4,
    "TURNOVER": 5,
    "FOUL": 6,
    "SUB": 8,
    "PERIOD_START": 12,
    "PERIOD_END": 13,
}

TEAM_NAMES = [
    ("LAL", "Lakers"), ("BOS", "Celtics"), ("GSW", "Warriors"), ("MIL", "Bucks"),
    ("PHX", "Suns"), ("DEN", "Nuggets"), ("MIA", "Heat"), ("NYK", "Knicks"),
]


def period_length_seconds(period: int) -> int:
    return 12 * 60 if period <= 4 else 5 * 60


def fmt_clock(seconds_left: int) -> str:
    m, s = divmod(max(seconds_left, 0), 60)
    return f"{m}:{s:02d}"


def simulate_game(game_id: str, home_abbr: str, away_abbr: str,
                   home_strength: float, away_strength: float,
                   rng: random.Random) -> tuple[pd.DataFrame, dict]:
    """
    Simulates one game possession-by-possession across 4 (or more) periods.
    `*_strength` loosely control scoring rate, [0.8, 1.2] is a sane range.
    Returns (play_by_play_df, label_dict).
    """
    events = []
    home_score, away_score = 0, 0
    home_fouls, away_fouls = 0, 0
    eventnum = 1
    period = 1
    possession = rng.choice(["HOME", "AWAY"])

    def add_event(period, seconds_left, msg_type, home_desc, away_desc):
        nonlocal eventnum
        margin = home_score - away_score
        events.append({
            "GAME_ID": game_id,
            "EVENTNUM": eventnum,
            "EVENTMSGTYPE": msg_type,
            "EVENTMSGACTIONTYPE": 0,
            "PERIOD": period,
            "WCTIMESTRING": "",
            "PCTIMESTRING": fmt_clock(seconds_left),
            "HOMEDESCRIPTION": home_desc,
            "NEUTRALDESCRIPTION": None,
            "VISITORDESCRIPTION": away_desc,
            "SCORE": f"{away_score} - {home_score}",
            "SCOREMARGIN": str(margin) if margin != 0 else "TIE",
            "HOME_FOULS": home_fouls,
            "AWAY_FOULS": away_fouls,
            "POSSESSION": possession,
        })
        eventnum += 1

    while period <= 4 or (home_score == away_score and period <= 6):
        seconds_left = period_length_seconds(period)
        add_event(period, seconds_left, EVENTMSGTYPE["PERIOD_START"], "Period Start", None)

        while seconds_left > 0:
            # possession length: 5-24 seconds
            possession_len = rng.randint(5, 24)
            seconds_left = max(0, seconds_left - possession_len)

            strength = home_strength if possession == "HOME" else away_strength
            roll = rng.random()

            if roll < 0.10 * strength:  # foul
                if possession == "HOME":
                    away_fouls += 1
                    add_event(period, seconds_left, EVENTMSGTYPE["FOUL"], None,
                              f"{away_abbr} Foul")
                else:
                    home_fouls += 1
                    add_event(period, seconds_left, EVENTMSGTYPE["FOUL"], f"{home_abbr} Foul", None)
            elif roll < 0.08 + 0.10:  # turnover, possession flips w/o scoring
                if possession == "HOME":
                    add_event(period, seconds_left, EVENTMSGTYPE["TURNOVER"], f"{home_abbr} Turnover", None)
                else:
                    add_event(period, seconds_left, EVENTMSGTYPE["TURNOVER"], None, f"{away_abbr} Turnover")
                possession = "AWAY" if possession == "HOME" else "HOME"
                continue
            else:
                make_prob = 0.45 * strength
                points = rng.choice([2, 2, 2, 3])
                if rng.random() < make_prob:
                    if possession == "HOME":
                        home_score += points
                        add_event(period, seconds_left, EVENTMSGTYPE["MADE_SHOT"],
                                  f"{home_abbr} made {points}pt shot", None)
                    else:
                        away_score += points
                        add_event(period, seconds_left, EVENTMSGTYPE["MADE_SHOT"],
                                  None, f"{away_abbr} made {points}pt shot")
                else:
                    if possession == "HOME":
                        add_event(period, seconds_left, EVENTMSGTYPE["MISSED_SHOT"],
                                  f"{home_abbr} missed shot", None)
                    else:
                        add_event(period, seconds_left, EVENTMSGTYPE["MISSED_SHOT"],
                                  None, f"{away_abbr} missed shot")
                    add_event(period, seconds_left, EVENTMSGTYPE["REBOUND"], "Rebound", "Rebound")

            possession = "AWAY" if possession == "HOME" else "HOME"

        add_event(period, 0, EVENTMSGTYPE["PERIOD_END"], "Period End", None)
        period += 1

    df = pd.DataFrame(events)
    label = {
        "GAME_ID": game_id,
        "HOME_TEAM_ID": home_abbr,
        "AWAY_TEAM_ID": away_abbr,
        "HOME_PTS": home_score,
        "AWAY_PTS": away_score,
        "HOME_WIN": int(home_score > away_score),
    }
    return df, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(RAW_DIR, exist_ok=True)

    all_pbp = []
    all_labels = []
    for i in range(args.n_games):
        (home_abbr, _), (away_abbr, _) = rng.sample(TEAM_NAMES, 2)
        home_strength = rng.uniform(0.85, 1.15)
        away_strength = rng.uniform(0.85, 1.15)
        game_id = f"SYN{i:05d}"
        df, label = simulate_game(game_id, home_abbr, away_abbr,
                                   home_strength, away_strength, rng)
        all_pbp.append(df)
        all_labels.append(label)

    pbp_all = pd.concat(all_pbp, ignore_index=True)
    labels_all = pd.DataFrame(all_labels)

    pbp_path = os.path.join(RAW_DIR, "play_by_play.csv")
    labels_path = os.path.join(RAW_DIR, "game_labels.csv")
    pbp_all.to_csv(pbp_path, index=False)
    labels_all.to_csv(labels_path, index=False)

    print(f"[generate_synthetic_data] {args.n_games} games simulated")
    print(f"[generate_synthetic_data] Saved {len(pbp_all)} events -> {pbp_path}")
    print(f"[generate_synthetic_data] Saved {len(labels_all)} game labels -> {labels_path}")


if __name__ == "__main__":
    main()
