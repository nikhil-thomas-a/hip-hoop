"""
build_windows.py
Builds before/after stat windows for each mention.

Key improvements:
- Off-season drops: baseline = last 30 games of PREV season,
  after windows = from start of NEXT season
- In-season drops: standard before/after split
- Uses all 3 fetched seasons so nothing is left on the table
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
GAME_LOG_DIR = RAW_DIR / "game_logs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STAT_COLS = ["PTS", "AST", "REB"]
OFFSEASON_GAP_DAYS = 45  # days gap = off-season


def load_game_log(player_name: str) -> pd.DataFrame | None:
    safe_name = player_name.lower().replace(" ", "_").replace("'", "")
    path = GAME_LOG_DIR / f"{safe_name}.csv"
    if not path.exists():
        print(f"  ✗ No game log found for: {player_name}")
        return None
    df = pd.read_csv(path)
    df["GAME_DATE_parsed"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.dropna(subset=["GAME_DATE_parsed"])
    df = df.sort_values("GAME_DATE_parsed").reset_index(drop=True)
    return df


def window_stats(games: pd.DataFrame) -> dict:
    if games.empty:
        return {f"{s}_mean": np.nan for s in STAT_COLS} | \
               {f"{s}_std": np.nan for s in STAT_COLS} | \
               {"n_games": 0}
    return {
        **{f"{s}_mean": round(float(games[s].mean()), 2) for s in STAT_COLS},
        **{f"{s}_std": round(float(games[s].std()), 2) for s in STAT_COLS},
        "n_games": len(games),
    }


def classify_drop(release_dt: pd.Timestamp, gl: pd.DataFrame) -> tuple[str, pd.Timestamp]:
    """
    Returns (drop_type, effective_date):
      'in_season'   — drop during active season, use release date
      'off_season'  — drop in off-season, shift to next season start
      'end_season'  — drop near playoff/end, shift to next season start
    """
    after = gl[gl["GAME_DATE_parsed"] >= release_dt]
    if after.empty:
        return "no_data", release_dt

    next_game = after.iloc[0]["GAME_DATE_parsed"]
    gap = (next_game - release_dt).days

    if gap <= OFFSEASON_GAP_DAYS:
        return "in_season", release_dt
    else:
        return "off_season", next_game


def build_windows(mentions_path=None) -> pd.DataFrame:
    if mentions_path is None:
        mentions_path = PROCESSED_DIR / "mentions_with_sentiment.csv"
        if not Path(mentions_path).exists():
            mentions_path = RAW_DIR / "mentions.csv"

    mentions = pd.read_csv(mentions_path)
    mentions["release_date_parsed"] = pd.to_datetime(mentions["release_date"])

    rows = []
    skipped = 0
    offseason_count = 0

    for _, mention in mentions.iterrows():
        player = mention["player"]
        release_dt = mention["release_date_parsed"]

        gl = load_game_log(player)
        if gl is None or gl.empty:
            skipped += 1
            continue

        drop_type, effective_dt = classify_drop(release_dt, gl)

        if drop_type == "no_data":
            print(f"  ⚠ No future games found for {player} after {release_dt.date()}")
            skipped += 1
            continue

        if drop_type == "off_season":
            offseason_count += 1
            # Baseline: last 30 games BEFORE the release date (end of prev season)
            before_games = gl[gl["GAME_DATE_parsed"] < release_dt].tail(30)
            # After windows: from start of NEXT season
            after_all = gl[gl["GAME_DATE_parsed"] >= effective_dt].reset_index(drop=True)
        else:
            # In-season: standard split
            before_games = gl[gl["GAME_DATE_parsed"] < release_dt].tail(30)
            after_all = gl[gl["GAME_DATE_parsed"] >= release_dt].reset_index(drop=True)

        if before_games.empty or after_all.empty:
            print(f"  ⚠ Insufficient data for {player} ({drop_type}) around {release_dt.date()}")
            skipped += 1
            continue

        after_1g     = after_all.head(1)
        after_10g    = after_all.head(10)
        after_30g    = after_all.head(30)
        after_season = after_all  # all games in the after window (rest of that season)

        row = {
            "mention_id":       mention["mention_id"],
            "player":           player,
            "artist":           mention["artist"],
            "song":             mention["song"],
            "release_date":     mention["release_date"],
            "mention_type":     mention["mention_type"],
            "artist_tier":      mention["artist_tier"],
            "player_position":  mention["player_position"],
            "era":              mention["era"],
            "drop_type":        drop_type,
            "effective_date":   str(effective_dt.date()),
        }

        for col in ["vader_compound", "vader_pos", "vader_neg"]:
            if col in mention.index:
                row[col] = mention[col]

        for prefix, games in [
            ("before",       before_games),
            ("after_1g",     after_1g),
            ("after_10g",    after_10g),
            ("after_30g",    after_30g),
            ("after_season", after_season),
        ]:
            stats = window_stats(games)
            for k, v in stats.items():
                row[f"{prefix}_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "windows.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Built {len(df)} mention windows ({offseason_count} off-season shifts, {skipped} skipped)")
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    build_windows()
