"""
fetch_lyrics.py
Fetches full lyrics for each song in mentions.csv via the Genius API.
Saves a lyrics_cache.json (gitignored) and a clean lyrics_sentiment.csv.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_FILE = RAW_DIR / "lyrics_cache.json"  # gitignored (can be large)
OUT_FILE = RAW_DIR / "lyrics_meta.csv"


def get_genius_client():
    token = os.getenv("GENIUS_API_KEY")
    if not token:
        raise EnvironmentError(
            "GENIUS_API_KEY not found. Copy .env.example to .env and add your key."
        )
    genius = lyricsgenius.Genius(
        token,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        remove_section_headers=True,
        timeout=15,
        retries=3,
        verbose=False,
    )
    return genius


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_all(mentions_path: str = None):
    if mentions_path is None:
        mentions_path = RAW_DIR / "mentions.csv"

    mentions = pd.read_csv(mentions_path)
    genius = get_genius_client()
    cache = load_cache()

    results = []
    unique_songs = mentions[["song", "artist", "mention_id", "lyric"]].drop_duplicates("mention_id")

    print(f"Fetching lyrics for {len(unique_songs)} songs...\n")

    for _, row in unique_songs.iterrows():
        key = f"{row['artist']}::{row['song']}"

        if key in cache:
            print(f"  ↩ {row['artist']} — {row['song']} (cached)")
            results.append({
                "mention_id": row["mention_id"],
                "song": row["song"],
                "artist": row["artist"],
                "lyric_snippet": row["lyric"],
                "full_lyrics_available": True,
                "lyrics_length": cache[key].get("length", 0),
                "url": cache[key].get("url", ""),
            })
            continue

        print(f"  → {row['artist']} — {row['song']}")
        try:
            song = genius.search_song(row["song"], row["artist"])
            if song:
                cache[key] = {
                    "title": song.title,
                    "artist": song.artist,
                    "url": song.url,
                    "length": len(song.lyrics) if song.lyrics else 0,
                }
                save_cache(cache)
                results.append({
                    "mention_id": row["mention_id"],
                    "song": row["song"],
                    "artist": row["artist"],
                    "lyric_snippet": row["lyric"],
                    "full_lyrics_available": True,
                    "lyrics_length": len(song.lyrics) if song.lyrics else 0,
                    "url": song.url,
                })
                print(f"  ✓ Found — {song.url}")
            else:
                print(f"  ✗ Not found on Genius")
                results.append({
                    "mention_id": row["mention_id"],
                    "song": row["song"],
                    "artist": row["artist"],
                    "lyric_snippet": row["lyric"],
                    "full_lyrics_available": False,
                    "lyrics_length": 0,
                    "url": "",
                })
            time.sleep(1.5)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            time.sleep(3)

    df = pd.DataFrame(results)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved lyrics metadata to {OUT_FILE}")
    return df


if __name__ == "__main__":
    fetch_all()
