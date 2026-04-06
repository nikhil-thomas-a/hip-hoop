"""
build_windows.py
For each mention, finds the player's game logs and slices them into
before/after windows relative to the song release date.

Windows produced:
  - before_30g : up to 30 games before release date
  - after_1g   : next 1 game
  - after_10g  : next ~10 games (approx 2 weeks)
  - after_30g  : next ~30 games (approx 1 month)
  - after_season: all remaining games that season

Output: data/processed/windows.csv — one row per mention, one col per stat/window.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
GAME_LOG_DIR = RAW_DIR / "game_logs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STAT_COLS = ["PTS", "AST", "REB"]


def load_game_log(player_name: str) -> pd.DataFrame | None:
    safe_name = player_name.lower().replace(" ", "_").replace("'", "")
    path = GAME_LOG_DIR / f"{safe_name}.csv"
    if not path.exists():
        print(f"  ✗ No game log found for: {player_name}")
        return None
    df = pd.read_csv(path)
    # nba_api returns GAME_DATE as 'MMM DD, YYYY'
    df["GAME_DATE_parsed"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.sort_values("GAME_DATE_parsed").reset_index(drop=True)
    return df


def window_stats(games: pd.DataFrame) -> dict:
    """Compute mean PTS, AST, REB for a set of games."""
    if games.empty:
        return {f"{s}_mean": np.nan for s in STAT_COLS} | \
               {f"{s}_std": np.nan for s in STAT_COLS} | \
               {"n_games": 0}
    return {
        **{f"{s}_mean": round(games[s].mean(), 2) for s in STAT_COLS},
        **{f"{s}_std": round(games[s].std(), 2) for s in STAT_COLS},
        "n_games": len(games),
    }


def find_effective_date(release_dt: pd.Timestamp, gl: pd.DataFrame) -> pd.Timestamp:
    """
    If the release date falls in the NBA off-season (no games within 45 days after),
    shift forward to the first game of the next season.
    This handles songs dropped in June/July/August/September.
    """
    nearby_after = gl[gl["GAME_DATE_parsed"] >= release_dt].head(1)
    if nearby_after.empty:
        return release_dt

    days_gap = (nearby_after.iloc[0]["GAME_DATE_parsed"] - release_dt).days
    if days_gap > 45:
        # Off-season drop — use the first game of the next season as event date
        # but keep the 30 games before the release date as baseline
        next_game_date = nearby_after.iloc[0]["GAME_DATE_parsed"]
        print(f"    ↪ Off-season drop ({release_dt.date()}) → shifted to {next_game_date.date()}")
        return next_game_date
    return release_dt


def build_windows(mentions_path: str = None) -> pd.DataFrame:
    if mentions_path is None:
        mentions_path = PROCESSED_DIR / "mentions_with_sentiment.csv"
        if not mentions_path.exists():
            mentions_path = RAW_DIR / "mentions.csv"

    mentions = pd.read_csv(mentions_path)
    mentions["release_date_parsed"] = pd.to_datetime(mentions["release_date"])

    rows = []
    skipped = 0

    for _, mention in mentions.iterrows():
        player = mention["player"]
        release_dt = mention["release_date_parsed"]

        gl = load_game_log(player)
        if gl is None or gl.empty:
            skipped += 1
            continue

        # Shift off-season dates to next season start
        effective_dt = find_effective_date(release_dt, gl)

        # Baseline: 30 games before the original release date
        before_mask = gl["GAME_DATE_parsed"] < release_dt
        # After windows: from effective date (handles off-season)
        after_mask  = gl["GAME_DATE_parsed"] >= effective_dt

        before_games = gl[before_mask].tail(30)        # up to 30 games before
        after_all    = gl[after_mask].reset_index(drop=True)
        after_1g     = after_all.head(1)
        after_10g    = after_all.head(10)
        after_30g    = after_all.head(30)
        after_season = after_all                        # rest of season

        if before_games.empty or after_all.empty:
            print(f"  ⚠ Insufficient data for {player} around {release_dt.date()}")
            skipped += 1
            continue

        row = {
            "mention_id":   mention["mention_id"],
            "player":       player,
            "artist":       mention["artist"],
            "song":         mention["song"],
            "release_date": mention["release_date"],
            "mention_type": mention["mention_type"],
            "artist_tier":  mention["artist_tier"],
            "player_position": mention["player_position"],
            "era":          mention["era"],
        }

        # Add sentiment if available
        for col in ["vader_compound", "vader_pos", "vader_neg"]:
            if col in mention.index:
                row[col] = mention[col]

        # Add windowed stats with prefixes
        for prefix, games in [
            ("before", before_games),
            ("after_1g", after_1g),
            ("after_10g", after_10g),
            ("after_30g", after_30g),
            ("after_season", after_season),
        ]:
            stats = window_stats(games)
            for k, v in stats.items():
                row[f"{prefix}_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "windows.csv"
    df.to_csv(out_path, index=False)
    print(f"\nBuilt {len(df)} mention windows (skipped {skipped})")
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    build_windows()
