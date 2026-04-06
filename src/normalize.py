"""
normalize.py
Adjusts raw stats for era and pace differences so that
1994 Shaq and 2022 Ja Morant are comparable.

Methods:
  1. Era z-score: normalize each stat within its era bucket
     (1990s / 2000s / 2010s / 2020s) using mean and std for that era.
  2. Pace adjustment: approximate points-per-100-possessions scaling.
     We use known average team pace per era as a proxy.

Adds *_normalized columns to windows.csv → normalized.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Average NBA pace (possessions per 48 min) per era — from Basketball Reference
ERA_PACE = {
    "1990s": 90.4,
    "2000s": 89.2,
    "2010s": 96.3,
    "2020s": 100.1,
}

# We normalize to 2010s as the reference era
REFERENCE_PACE = ERA_PACE["2010s"]

STAT_COLS = ["PTS", "AST", "REB"]
WINDOWS = ["before", "after_1g", "after_10g", "after_30g", "after_season"]


def pace_factor(era: str) -> float:
    """Scale factor to adjust era stats to reference era pace."""
    return REFERENCE_PACE / ERA_PACE.get(era, REFERENCE_PACE)


def normalize(windows_path: str = None) -> pd.DataFrame:
    if windows_path is None:
        windows_path = PROCESSED_DIR / "windows.csv"

    df = pd.read_csv(windows_path)

    # --- 1. Pace-adjust all mean stats ---
    for window in WINDOWS:
        for stat in STAT_COLS:
            col = f"{window}_{stat}_mean"
            if col in df.columns:
                df[f"{window}_{stat}_pace_adj"] = df.apply(
                    lambda row: round(row[col] * pace_factor(row["era"]), 2)
                    if pd.notna(row[col]) else np.nan,
                    axis=1,
                )

    # --- 2. Compute stat deltas (after minus before) ---
    for window in ["after_1g", "after_10g", "after_30g", "after_season"]:
        for stat in STAT_COLS:
            before_col = f"before_{stat}_mean"
            after_col  = f"{window}_{stat}_mean"
            adj_before  = f"before_{stat}_pace_adj"
            adj_after   = f"{window}_{stat}_pace_adj"

            if before_col in df.columns and after_col in df.columns:
                df[f"delta_{window}_{stat}"] = round(df[after_col] - df[before_col], 2)

            if adj_before in df.columns and adj_after in df.columns:
                df[f"delta_{window}_{stat}_adj"] = round(df[adj_after] - df[adj_before], 2)

    # --- 3. Era z-score normalization ---
    for window in ["after_30g"]:  # z-score the primary analysis window
        for stat in STAT_COLS:
            delta_col = f"delta_{window}_{stat}_adj"
            if delta_col in df.columns:
                era_groups = df.groupby("era")[delta_col]
                df[f"zscore_{window}_{stat}"] = era_groups.transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                ).round(3)

    # --- 4. Composite impact score per mention ---
    # Weighted: PTS (40%) + AST (30%) + REB (30%), pace-adjusted, 30-game window
    def composite_delta(row):
        pts = row.get("delta_after_30g_PTS_adj", np.nan)
        ast = row.get("delta_after_30g_AST_adj", np.nan)
        reb = row.get("delta_after_30g_REB_adj", np.nan)
        if any(pd.isna([pts, ast, reb])):
            return np.nan
        return round(0.4 * pts + 0.3 * ast + 0.3 * reb, 3)

    df["composite_delta_30g"] = df.apply(composite_delta, axis=1)

    # Tier-weighted version
    df["weighted_impact"] = (df["composite_delta_30g"] * df["artist_tier"]).round(3)

    out_path = PROCESSED_DIR / "normalized.csv"
    df.to_csv(out_path, index=False)
    print(f"Normalized data saved to {out_path}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    normalize()
