"""
features.py
------------
Turns raw play-by-play events into the 4 model features called out in the
video's "Model Features" card:

    FEAT 1  Score Differential   -> current point margin (home - away)
    FEAT 2  Time Remaining       -> seconds left in game (regulation + OT)
    FEAT 3  Possession           -> which team currently has the ball (0/1)
    FEAT 4  Fouls                -> team foul count + bonus flag

Works against either:
    - real nba_api PlayByPlayV2 output (parses HOMEDESCRIPTION /
      VISITORDESCRIPTION text to infer possession + fouls), or
    - the synthetic data from generate_synthetic_data.py (which already
      carries POSSESSION / HOME_FOULS / AWAY_FOULS columns directly)

Output: one row per event, one game per group, with columns:
    game_id, score_diff, seconds_remaining, possession_home, home_fouls,
    away_fouls, home_bonus, away_bonus, label (home_win, broadcast to every
    row of that game so we can train on every snapshot in time)
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

REGULATION_PERIOD_SEC = 12 * 60
OT_PERIOD_SEC = 5 * 60
FOUL_BONUS_THRESHOLD = 5  # team fouls in a period that trigger bonus FTs


def _period_seconds_remaining_total(period: int, pctimestring: str) -> int:
    """Seconds remaining in the *entire game* (not just the period)."""
    try:
        m, s = pctimestring.split(":")
        secs_left_in_period = int(m) * 60 + int(s)
    except Exception:
        secs_left_in_period = 0

    if period <= 4:
        periods_after_this = 4 - period
        return secs_left_in_period + periods_after_this * REGULATION_PERIOD_SEC
    else:
        # overtime: we don't know how many more OTs there will be, so we
        # just report time left in the current OT period as a reasonable
        # proxy (win-probability models typically clip / rescale this).
        return secs_left_in_period


def _parse_score_margin(row) -> int:
    """home - away, parsed from the SCORE 'away - home' string."""
    score = row.get("SCORE")
    if isinstance(score, str) and "-" in score:
        try:
            away, home = [int(x.strip()) for x in score.split("-")]
            return home - away
        except Exception:
            pass
    margin = row.get("SCOREMARGIN")
    if isinstance(margin, str):
        if margin == "TIE":
            return 0
        try:
            return int(margin)
        except Exception:
            return 0
    return 0


def build_features_for_game(game_df: pd.DataFrame, home_win: int) -> pd.DataFrame:
    game_df = game_df.sort_values("EVENTNUM").reset_index(drop=True)
    has_synthetic_cols = {"POSSESSION", "HOME_FOULS", "AWAY_FOULS"}.issubset(game_df.columns)

    rows = []
    running_score = 0  # forward-filled home-away margin
    home_fouls_running = 0
    away_fouls_running = 0
    last_period = 1
    possession = "HOME"

    for _, ev in game_df.iterrows():
        period = int(ev["PERIOD"])
        if period != last_period:
            # team fouls reset each period in real NBA rules
            home_fouls_running = 0
            away_fouls_running = 0
            last_period = period

        margin = _parse_score_margin(ev)
        if margin != 0 or isinstance(ev.get("SCORE"), str):
            running_score = margin

        if has_synthetic_cols:
            possession = ev["POSSESSION"]
            home_fouls_running = int(ev["HOME_FOULS"])
            away_fouls_running = int(ev["AWAY_FOULS"])
        else:
            home_desc = ev.get("HOMEDESCRIPTION")
            away_desc = ev.get("VISITORDESCRIPTION")
            if isinstance(home_desc, str):
                possession = "HOME"
                if "Foul" in home_desc:
                    home_fouls_running += 1
            elif isinstance(away_desc, str):
                possession = "AWAY"
                if "Foul" in away_desc:
                    away_fouls_running += 1

        seconds_remaining = _period_seconds_remaining_total(period, ev.get("PCTIMESTRING", "0:00"))

        rows.append({
            "game_id": ev["GAME_ID"],
            "eventnum": ev["EVENTNUM"],
            "score_diff": running_score,
            "seconds_remaining": seconds_remaining,
            "possession_home": 1 if possession == "HOME" else 0,
            "home_fouls": home_fouls_running,
            "away_fouls": away_fouls_running,
            "home_bonus": int(home_fouls_running >= FOUL_BONUS_THRESHOLD),
            "away_bonus": int(away_fouls_running >= FOUL_BONUS_THRESHOLD),
            "label": home_win,
        })

    return pd.DataFrame(rows)


def build_dataset(pbp_path: str = None, labels_path: str = None) -> pd.DataFrame:
    pbp_path = pbp_path or os.path.join(RAW_DIR, "play_by_play.csv")
    labels_path = labels_path or os.path.join(RAW_DIR, "game_labels.csv")

    pbp = pd.read_csv(pbp_path)
    labels = pd.read_csv(labels_path).set_index("GAME_ID")["HOME_WIN"].to_dict()

    all_features = []
    for game_id, game_df in pbp.groupby("GAME_ID"):
        home_win = labels.get(game_id)
        if home_win is None:
            continue
        feats = build_features_for_game(game_df, home_win)
        all_features.append(feats)

    dataset = pd.concat(all_features, ignore_index=True)
    return dataset


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dataset = build_dataset()
    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    dataset.to_csv(out_path, index=False)
    print(f"[features] Built {len(dataset)} rows across "
          f"{dataset['game_id'].nunique()} games -> {out_path}")
    print(dataset.head())


if __name__ == "__main__":
    main()
