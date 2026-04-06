"""
export.py
Final step: reads normalized.csv + statistical outputs and
assembles results.json — the single file consumed by index.html.

Structure of results.json:
{
  "generated_at": "...",
  "n_mentions": 110,
  "key_findings": { ... },
  "statistical_tests": { ... },
  "regression": { ... },
  "correlations": { ... },
  "mentions": [ ... ],  // full detail per mention
  "top_impacts": [ ... ],
  "era_breakdown": { ... },
  "type_breakdown": { ... },
  "position_breakdown": { ... }
}
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_FILE = Path(__file__).parent.parent / "data" / "processed" / "results.json"


class NpEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 4) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)


def safe_round(val, n=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), n)


def paired_ttest(before_col, after_col, df):
    """Run a paired t-test, returning t-stat, p-value, and effect size (Cohen's d)."""
    valid = df[[before_col, after_col]].dropna()
    if len(valid) < 5:
        return {"t": None, "p": None, "cohens_d": None, "n": len(valid), "significant": False}
    t, p = stats.ttest_rel(valid[after_col], valid[before_col])
    diff = valid[after_col] - valid[before_col]
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    return {
        "t": safe_round(t),
        "p": safe_round(p, 4),
        "cohens_d": safe_round(d),
        "mean_delta": safe_round(diff.mean()),
        "n": len(valid),
        "significant": bool(p < 0.05),
    }


def correlation_stats(x_col, y_col, df, label=""):
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 5:
        return None
    r, p = stats.pearsonr(valid[x_col], valid[y_col])
    return {
        "variable": label or y_col,
        "r": safe_round(r),
        "p": safe_round(p, 4),
        "n": len(valid),
        "significant": bool(p < 0.05),
    }


def build_export():
    norm_path = PROCESSED_DIR / "normalized.csv"
    if not norm_path.exists():
        raise FileNotFoundError(
            "normalized.csv not found. Run notebooks 01–03 first."
        )

    df = pd.read_csv(norm_path)
    df_valid = df[df["delta_after_30g_PTS_adj"].notna()].copy()

    print(f"Exporting {len(df)} total mentions, {len(df_valid)} with complete data...")

    # ── KEY FINDINGS ──────────────────────────────────────────
    avg_pts_delta  = safe_round(df_valid["delta_after_30g_PTS_adj"].mean(), 2)
    avg_ast_delta  = safe_round(df_valid["delta_after_30g_AST_adj"].mean(), 2)
    avg_reb_delta  = safe_round(df_valid["delta_after_30g_REB_adj"].mean(), 2)
    pct_positive   = safe_round((df_valid["composite_delta_30g"] > 0).mean() * 100, 1)
    strongest_idx  = df_valid["composite_delta_30g"].abs().idxmax() if not df_valid.empty else None

    key_findings = {
        "avg_pts_delta_30g":  avg_pts_delta,
        "avg_ast_delta_30g":  avg_ast_delta,
        "avg_reb_delta_30g":  avg_reb_delta,
        "pct_positive_impact": pct_positive,
        "n_total":   len(df),
        "n_analyzed": len(df_valid),
        "n_compliment": int((df["mention_type"] == "compliment").sum()),
        "n_diss":       int((df["mention_type"] == "diss").sum()),
        "n_neutral":    int((df["mention_type"] == "neutral").sum()),
    }

    # ── STATISTICAL TESTS ─────────────────────────────────────
    statistical_tests = {}
    for stat in ["PTS", "AST", "REB"]:
        statistical_tests[stat] = {
            "30g": paired_ttest(f"before_{stat}_mean", f"after_30g_{stat}_mean", df_valid),
            "10g": paired_ttest(f"before_{stat}_mean", f"after_10g_{stat}_mean", df_valid),
            "season": paired_ttest(f"before_{stat}_mean", f"after_season_{stat}_mean", df_valid),
        }

    # By type
    type_tests = {}
    for t in ["compliment", "diss", "neutral"]:
        sub = df_valid[df_valid["mention_type"] == t]
        type_tests[t] = {
            stat: paired_ttest(f"before_{stat}_mean", f"after_30g_{stat}_mean", sub)
            for stat in ["PTS", "AST", "REB"]
        }
    statistical_tests["by_type"] = type_tests

    # ── CORRELATIONS ─────────────────────────────────────────
    correlations = []
    if "vader_compound" in df_valid.columns:
        for stat in ["PTS", "AST", "REB"]:
            c = correlation_stats("vader_compound", f"delta_after_30g_{stat}_adj", df_valid,
                                  f"Sentiment → Δ{stat}")
            if c:
                correlations.append(c)

    for stat in ["PTS", "AST", "REB"]:
        c = correlation_stats("artist_tier", f"delta_after_30g_{stat}_adj", df_valid,
                              f"Artist Tier → Δ{stat}")
        if c:
            correlations.append(c)

    # Tier × Sentiment interaction
    if "vader_compound" in df_valid.columns:
        df_valid["tier_x_sentiment"] = df_valid["artist_tier"] * df_valid["vader_compound"]
        c = correlation_stats("tier_x_sentiment", "composite_delta_30g", df_valid,
                              "Tier × Sentiment → Composite Δ")
        if c:
            correlations.append(c)

    # ── BREAKDOWNS ────────────────────────────────────────────
    def breakdown(group_col):
        if group_col not in df_valid.columns:
            return {}
        return {
            str(k): {
                "n": int(v["composite_delta_30g"].count()),
                "mean_composite_delta": safe_round(v["composite_delta_30g"].mean(), 2),
                "mean_pts_delta": safe_round(v["delta_after_30g_PTS_adj"].mean(), 2),
                "mean_ast_delta": safe_round(v["delta_after_30g_AST_adj"].mean(), 2),
                "mean_reb_delta": safe_round(v["delta_after_30g_REB_adj"].mean(), 2),
            }
            for k, v in df_valid.groupby(group_col)
        }

    # ── PER-MENTION RECORDS ────────────────────────────────────
    mention_records = []
    for _, row in df.iterrows():
        rec = {
            "mention_id":    int(row["mention_id"]),
            "player":        row["player"],
            "artist":        row["artist"],
            "song":          row["song"],
            "release_date":  row["release_date"],
            "mention_type":  row["mention_type"],
            "artist_tier":   safe_round(row["artist_tier"], 1),
            "era":           row["era"],
            "position":      row["player_position"],
            "lyric":         "",  # populated from raw mentions below
            "stats": {
                "before": {
                    "pts": safe_round(row.get("before_PTS_mean"), 1),
                    "ast": safe_round(row.get("before_AST_mean"), 1),
                    "reb": safe_round(row.get("before_REB_mean"), 1),
                    "n":   safe_round(row.get("before_n_games"), 0),
                },
                "after_1g": {
                    "pts": safe_round(row.get("after_1g_PTS_mean"), 1),
                    "ast": safe_round(row.get("after_1g_AST_mean"), 1),
                    "reb": safe_round(row.get("after_1g_REB_mean"), 1),
                },
                "after_10g": {
                    "pts": safe_round(row.get("after_10g_PTS_mean"), 1),
                    "ast": safe_round(row.get("after_10g_AST_mean"), 1),
                    "reb": safe_round(row.get("after_10g_REB_mean"), 1),
                },
                "after_30g": {
                    "pts": safe_round(row.get("after_30g_PTS_mean"), 1),
                    "ast": safe_round(row.get("after_30g_AST_mean"), 1),
                    "reb": safe_round(row.get("after_30g_REB_mean"), 1),
                },
                "after_season": {
                    "pts": safe_round(row.get("after_season_PTS_mean"), 1),
                    "ast": safe_round(row.get("after_season_AST_mean"), 1),
                    "reb": safe_round(row.get("after_season_REB_mean"), 1),
                },
            },
            "deltas": {
                "pts_30g": safe_round(row.get("delta_after_30g_PTS_adj"), 2),
                "ast_30g": safe_round(row.get("delta_after_30g_AST_adj"), 2),
                "reb_30g": safe_round(row.get("delta_after_30g_REB_adj"), 2),
                "composite_30g": safe_round(row.get("composite_delta_30g"), 3),
                "weighted_impact": safe_round(row.get("weighted_impact"), 3),
            },
            "sentiment": {
                "vader_compound": safe_round(row.get("vader_compound"), 3),
            } if "vader_compound" in row.index else {},
        }
        mention_records.append(rec)

    # Inject lyric text from raw mentions
    raw_mentions = pd.read_csv(PROCESSED_DIR.parent / "raw" / "mentions.csv")
    lyric_map = dict(zip(raw_mentions["mention_id"], raw_mentions["lyric"]))
    for rec in mention_records:
        rec["lyric"] = lyric_map.get(rec["mention_id"], "")

    # Top 10 by absolute impact
    valid_recs = [r for r in mention_records if r["deltas"]["composite_30g"] is not None]
    top_impacts = sorted(valid_recs, key=lambda r: abs(r["deltas"]["composite_30g"] or 0), reverse=True)[:10]

    # ── ASSEMBLE FINAL JSON ────────────────────────────────────
    results = {
        "generated_at": datetime.now().isoformat(),
        "n_mentions": len(df),
        "key_findings": key_findings,
        "statistical_tests": statistical_tests,
        "correlations": correlations,
        "era_breakdown": breakdown("era"),
        "type_breakdown": breakdown("mention_type"),
        "position_breakdown": breakdown("player_position"),
        "tier_breakdown": breakdown("artist_tier"),
        "mentions": mention_records,
        "top_impacts": top_impacts,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=2)

    print(f"\n✓ results.json written to {OUT_FILE}")
    print(f"  {len(mention_records)} mention records")
    print(f"  Key findings: {key_findings}")
    return results


if __name__ == "__main__":
    build_export()
