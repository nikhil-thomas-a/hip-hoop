"""
sentiment.py
Applies VADER sentiment analysis to each lyric snippet in mentions.csv.
Produces a continuous sentiment score (-1 to +1) per mention,
which is more nuanced than the binary compliment/diss label.
"""

import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def score_lyrics(mentions_path: str = None) -> pd.DataFrame:
    if mentions_path is None:
        mentions_path = RAW_DIR / "mentions.csv"

    df = pd.read_csv(mentions_path)
    analyzer = SentimentIntensityAnalyzer()

    scores = []
    for _, row in df.iterrows():
        lyric = str(row["lyric"])
        vs = analyzer.polarity_scores(lyric)
        scores.append({
            "mention_id": row["mention_id"],
            "vader_compound": vs["compound"],   # -1 (most negative) to +1 (most positive)
            "vader_pos": vs["pos"],
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
        })

    sentiment_df = pd.DataFrame(scores)
    df = df.merge(sentiment_df, on="mention_id")

    out_path = PROCESSED_DIR / "mentions_with_sentiment.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sentiment scores to {out_path}")
    print(f"\nSentiment summary:")
    print(df["vader_compound"].describe().round(3))
    print(f"\nMean by mention_type:")
    print(df.groupby("mention_type")["vader_compound"].mean().round(3))

    return df


if __name__ == "__main__":
    score_lyrics()
