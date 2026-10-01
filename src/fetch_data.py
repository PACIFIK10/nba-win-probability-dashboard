"""
fetch_data.py
-------------
Pulls play-by-play logs for a list of NBA game IDs using nba_api and caches
them to disk as raw CSVs. This is the "DATA" layer of the tech stack
(nba_api -> play-by-play logs).

Run standalone:
    python src/fetch_data.py --season 2023-24 --n-games 25

Notes:
    - nba_api talks to stats.nba.com. That endpoint is NOT reachable from
      network-restricted sandboxes (it isn't on most allowlists), so this
      script is meant to be run on your own machine / a normal dev box.
    - For development without network access, use
      `src/generate_synthetic_data.py` instead, which produces data in the
      exact same schema so the rest of the pipeline (features -> model ->
      dashboard) can be built and tested offline.
"""
import argparse
import os
import time
import sys

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_game_ids(season: str, n_games: int) -> list[str]:
    """Grab a list of finished game IDs for a season using LeagueGameFinder."""
    from nba_api.stats.endpoints import leaguegamefinder

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
    )
    games = finder.get_data_frames()[0]
    # Each game appears twice (once per team) - dedupe on GAME_ID
    game_ids = games["GAME_ID"].drop_duplicates().tolist()
    return game_ids[:n_games]


def fetch_play_by_play(game_id: str) -> pd.DataFrame:
    """Fetch raw play-by-play for a single game."""
    from nba_api.stats.endpoints import playbyplayv2

    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
    df = pbp.get_data_frames()[0]
    df["GAME_ID"] = game_id
    return df


def fetch_boxscore_summary(game_id: str) -> dict:
    """Fetch final score / winner for a game, used as the training label."""
    from nba_api.stats.endpoints import boxscoresummaryv2

    box = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
    line_score = box.get_data_frames()[1]
    home_row = line_score.iloc[1]
    away_row = line_score.iloc[0]
    return {
        "GAME_ID": game_id,
        "HOME_TEAM_ID": home_row["TEAM_ID"],
        "AWAY_TEAM_ID": away_row["TEAM_ID"],
        "HOME_PTS": home_row["PTS"],
        "AWAY_PTS": away_row["PTS"],
        "HOME_WIN": int(home_row["PTS"] > away_row["PTS"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2023-24", help="e.g. 2023-24")
    parser.add_argument("--n-games", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.6,
                         help="seconds between requests (be nice to stats.nba.com)")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"[fetch_data] Looking up game IDs for season {args.season} ...")
    try:
        game_ids = get_game_ids(args.season, args.n_games)
    except Exception as e:
        print(f"[fetch_data] ERROR: could not reach stats.nba.com ({e})")
        print("[fetch_data] If you're offline / sandboxed, run "
              "src/generate_synthetic_data.py instead.")
        sys.exit(1)

    print(f"[fetch_data] Found {len(game_ids)} games. Fetching play-by-play + labels...")

    all_pbp = []
    all_labels = []
    for i, gid in enumerate(game_ids, 1):
        try:
            pbp_df = fetch_play_by_play(gid)
            label = fetch_boxscore_summary(gid)
            all_pbp.append(pbp_df)
            all_labels.append(label)
            print(f"  [{i}/{len(game_ids)}] game {gid}: {len(pbp_df)} events, "
                  f"home_win={label['HOME_WIN']}")
        except Exception as e:
            print(f"  [{i}/{len(game_ids)}] game {gid}: FAILED ({e})")
        time.sleep(args.sleep)

    if not all_pbp:
        print("[fetch_data] No games fetched successfully. Exiting.")
        sys.exit(1)

    pbp_all = pd.concat(all_pbp, ignore_index=True)
    labels_all = pd.DataFrame(all_labels)

    pbp_path = os.path.join(RAW_DIR, "play_by_play.csv")
    labels_path = os.path.join(RAW_DIR, "game_labels.csv")
    pbp_all.to_csv(pbp_path, index=False)
    labels_all.to_csv(labels_path, index=False)

    print(f"[fetch_data] Saved {len(pbp_all)} events -> {pbp_path}")
    print(f"[fetch_data] Saved {len(labels_all)} game labels -> {labels_path}")


if __name__ == "__main__":
    main()
