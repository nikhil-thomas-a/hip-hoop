"""
fetch_gamelogs.py
Pulls per-game stats for each player in mentions.csv using nba_api.
Saves one CSV per player to data/raw/game_logs/{player_id}.csv
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

# Map Basketball Reference IDs → NBA.com player names for lookup
# nba_api uses its own IDs — we look up by name

# Known name differences between our CSV and nba_api
NAME_ALIASES = {
    "amare stoudemire":          "Amar'e Stoudemire",
    "nikola jokic":              "Nikola Jokic",
    "luka doncic":               "Luka Doncic",
    "de'aaron fox":              "De'Aaron Fox",
    "donovan mitchell":          "Donovan Mitchell",
    "domantas sabonis":          "Domantas Sabonis",
    "jimmy butler":              "Jimmy Butler",
    "pascal siakam":             "Pascal Siakam",
    "scottie barnes":            "Scottie Barnes",
    "cade cunningham":           "Cade Cunningham",
    "jalen green":               "Jalen Green",
    "anthony edwards":           "Anthony Edwards",
    "lamelo ball":               "LaMelo Ball",
    "penny hardaway":            "Anfernee Hardaway",
    "anfernee hardaway":         "Anfernee Hardaway",
}

def get_nba_api_id(player_name: str) -> int | None:
    all_players = players.get_players()

    # Check alias map first
    lookup_name = NAME_ALIASES.get(player_name.lower(), player_name)

    # Exact match
    matches = [p for p in all_players if p["full_name"].lower() == lookup_name.lower()]
    if matches:
        return matches[0]["id"]

    # Last name match (handles "Nikola Jokić" accent variants)
    last_name = lookup_name.split()[-1].lower()
    matches = [p for p in all_players if p["last_name"].lower() == last_name]
    if len(matches) == 1:
        return matches[0]["id"]

    # Partial match fallback
    matches = [p for p in all_players if lookup_name.lower() in p["full_name"].lower()]
    if matches:
        return matches[0]["id"]

    return None


def fetch_player_seasons(player_name: str, seasons: list[str]) -> pd.DataFrame:
    """Fetch game logs for a player across multiple seasons."""
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
            df["SEASON"] = season
            df["PLAYER_NAME"] = player_name
            frames.append(df)
            time.sleep(0.7)  # be polite to the API
        except Exception as e:
            print(f"  ✗ Error fetching {player_name} {season}: {e}")
            time.sleep(2)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def season_from_date(release_date: str) -> list[str]:
    """Return the two NBA seasons surrounding a release date."""
    year = int(release_date[:4])
    month = int(release_date[5:7])
    # NBA season starts October — if release is Oct-Dec, it's in the new season
    if month >= 10:
        season_year = year
    else:
        season_year = year - 1
    # Return prior season + current season
    def fmt(y):
        return f"{y}-{str(y+1)[-2:]}"
    return [fmt(season_year - 1), fmt(season_year)]


def fetch_all(mentions_path: str = None):
    if mentions_path is None:
        mentions_path = RAW_DIR / "mentions.csv"

    mentions = pd.read_csv(mentions_path)
    unique_players = mentions[["player", "release_date"]].drop_duplicates("player")

    print(f"Fetching game logs for {len(unique_players)} unique players...\n")

    for _, row in tqdm(unique_players.iterrows(), total=len(unique_players)):
        player_name = row["player"]
        release_date = row["release_date"]
        safe_name = player_name.lower().replace(" ", "_").replace("'", "")
        out_path = GAME_LOG_DIR / f"{safe_name}.csv"

        if out_path.exists():
            print(f"  ↩ Skipping {player_name} (already fetched)")
            continue

        print(f"  → Fetching {player_name}...")
        seasons = season_from_date(release_date)

        # Some players appear multiple times with different release dates
        all_dates = mentions[mentions["player"] == player_name]["release_date"].tolist()
        extra_seasons = set()
        for d in all_dates:
            for s in season_from_date(d):
                extra_seasons.add(s)
        all_seasons = list(set(seasons) | extra_seasons)

        df = fetch_player_seasons(player_name, all_seasons)

        if not df.empty:
            df.to_csv(out_path, index=False)
            print(f"  ✓ Saved {len(df)} games for {player_name}")
        else:
            print(f"  ✗ No data for {player_name}")

        time.sleep(0.5)

    print("\nDone fetching game logs.")


if __name__ == "__main__":
    fetch_all()
