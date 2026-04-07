"""
fetch_gamelogs.py
Pulls per-game stats for each player in mentions.csv using nba_api.
Fetches 3 seasons per mention (prev, current, next) so off-season
drops always have a next-season window to analyze.
Saves one CSV per player to data/raw/game_logs/{player_name}.csv
"""

import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
GAME_LOG_DIR = RAW_DIR / "game_logs"
GAME_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Known name differences between our CSV and nba_api
NAME_ALIASES = {
    "amare stoudemire":     "Amar'e Stoudemire",
    "nikola jokic":         "Nikola Jokic",
    "luka doncic":          "Luka Doncic",
    "de'aaron fox":         "De'Aaron Fox",
    "donovan mitchell":     "Donovan Mitchell",
    "domantas sabonis":     "Domantas Sabonis",
    "jimmy butler":         "Jimmy Butler",
    "pascal siakam":        "Pascal Siakam",
    "scottie barnes":       "Scottie Barnes",
    "cade cunningham":      "Cade Cunningham",
    "jalen green":          "Jalen Green",
    "anthony edwards":      "Anthony Edwards",
    "lamelo ball":          "LaMelo Ball",
    "penny hardaway":       "Anfernee Hardaway",
    "anfernee hardaway":    "Anfernee Hardaway",
    "kevin johnson":        "Kevin Johnson",
    "muggsy bogues":        "Muggsy Bogues",
    "mitch richmond":       "Mitch Richmond",
    "nick van exel":        "Nick Van Exel",
    "latrell sprewell":     "Latrell Sprewell",
    "glen rice":            "Glen Rice",
    "shawn kemp":           "Shawn Kemp",
}


def get_nba_api_id(player_name: str) -> int | None:
    all_players = players.get_players()
    lookup_name = NAME_ALIASES.get(player_name.lower(), player_name)

    # Exact match
    matches = [p for p in all_players if p["full_name"].lower() == lookup_name.lower()]
    if matches:
        return matches[0]["id"]

    # Last name match (handles accent variants like Jokić)
    last_name = lookup_name.split()[-1].lower()
    matches = [p for p in all_players if p["last_name"].lower() == last_name]
    if len(matches) == 1:
        return matches[0]["id"]

    # Partial match fallback
    matches = [p for p in all_players if lookup_name.lower() in p["full_name"].lower()]
    if matches:
        return matches[0]["id"]

    return None


def season_fmt(y: int) -> str:
    return f"{y}-{str(y+1)[-2:]}"


def seasons_for_date(release_date: str) -> list[str]:
    """
    Return 3 seasons surrounding a release date:
    prev season, current season, next season.
    This ensures off-season drops (Jun-Sep) always have 
    a full next-season window available.
    """
    year = int(release_date[:4])
    month = int(release_date[5:7])
    # NBA season starts in October
    # If drop is Oct-Dec → current season starts that year
    # If drop is Jan-Sep → current season started prior year
    if month >= 10:
        season_year = year
    else:
        season_year = year - 1

    return [
        season_fmt(season_year - 1),  # season before
        season_fmt(season_year),       # season of the drop
        season_fmt(season_year + 1),   # season after (crucial for off-season drops)
    ]


def fetch_player_seasons(player_name: str, seasons: list[str]) -> pd.DataFrame:
    nba_id = get_nba_api_id(player_name)
    if nba_id is None:
        print(f"  ✗ Could not find NBA API ID for: {player_name}")
        return pd.DataFrame()

    frames = []
    for season in seasons:
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=nba_id,
                season=season,
                season_type_all_star="Regular Season",
                timeout=60,
            )
            df = gl.get_data_frames()[0]
            if not df.empty:
                df["SEASON"] = season
                df["PLAYER_NAME"] = player_name
                frames.append(df)
            time.sleep(0.6)
        except Exception as e:
            print(f"  ✗ Error fetching {player_name} {season}: {e}")
            time.sleep(2)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_all(mentions_path: str = None):
    if mentions_path is None:
        mentions_path = RAW_DIR / "mentions.csv"

    mentions = pd.read_csv(mentions_path)
    # Get all seasons needed per player across all their mentions
    player_seasons = {}
    for _, row in mentions.iterrows():
        p = row["player"]
        seasons = set(seasons_for_date(row["release_date"]))
        if p not in player_seasons:
            player_seasons[p] = set()
        player_seasons[p] |= seasons

    print(f"Fetching game logs for {len(player_seasons)} unique players...\n")

    for player_name, seasons in tqdm(player_seasons.items()):
        safe_name = player_name.lower().replace(" ", "_").replace("'", "")
        out_path = GAME_LOG_DIR / f"{safe_name}.csv"

        # Check if we already have enough seasons
        if out_path.exists():
            existing = pd.read_csv(out_path)
            if "SEASON" in existing.columns:
                existing_seasons = set(existing["SEASON"].unique())
                missing = seasons - existing_seasons
                if not missing:
                    continue  # all seasons present
                print(f"  ↩ Updating {player_name} — fetching missing seasons: {missing}")
                new_df = fetch_player_seasons(player_name, list(missing))
                if not new_df.empty:
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.drop_duplicates(subset=["Game_ID"] if "Game_ID" in combined.columns else None,
                                             inplace=True)
                    combined.to_csv(out_path, index=False)
                continue

        print(f"  → Fetching {player_name} ({len(seasons)} seasons)...")
        df = fetch_player_seasons(player_name, sorted(seasons))
        if not df.empty:
            df.to_csv(out_path, index=False)
            print(f"  ✓ {len(df)} games for {player_name}")
        else:
            print(f"  ✗ No data for {player_name}")
        time.sleep(0.4)

    print("\nDone fetching game logs.")


if __name__ == "__main__":
    fetch_all()
