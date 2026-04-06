"""
Generates all 5 Jupyter notebooks for the Hip Hoop pipeline.
Run: python generate_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ─────────────────────────────────────────────────────────────
# NOTEBOOK 01 — Data Collection
# ─────────────────────────────────────────────────────────────
nb01 = nb([
    md("# 01 · Data Collection\nFetches NBA game logs and Genius lyrics metadata for all 110 mentions.\n\n**Run time:** ~15–25 minutes (API rate limits).\n\nPrerequisites:\n```\npip install -r requirements.txt\ncp .env.example .env  # then add your GENIUS_API_KEY\n```"),
    code("""\
import sys
sys.path.insert(0, '..')
from src.fetch_gamelogs import fetch_all as fetch_games
from src.fetch_lyrics import fetch_all as fetch_lyrics
import pandas as pd

mentions = pd.read_csv('../data/raw/mentions.csv')
print(f"Loaded {len(mentions)} mentions across {mentions['player'].nunique()} unique players")
mentions.head()
"""),
    code("""\
# Fetch NBA game logs (nba_api)
# This hits NBA's API — be patient, it takes a few minutes
# Already-fetched players are skipped automatically
fetch_games('../data/raw/mentions.csv')
"""),
    code("""\
# Verify game logs downloaded
import os
log_dir = '../data/raw/game_logs'
logs = os.listdir(log_dir)
print(f"Game logs fetched: {len(logs)}")
print("\\n".join(sorted(logs)[:10]), "...")
"""),
    code("""\
# Fetch Genius lyrics metadata
# Requires GENIUS_API_KEY in .env
fetch_lyrics('../data/raw/mentions.csv')
"""),
    code("""\
# Quick sanity check on a single game log
import pandas as pd
sample = pd.read_csv('../data/raw/game_logs/lebron_james.csv')
print(f"LeBron game log: {len(sample)} games")
print(sample[['GAME_DATE', 'PTS', 'AST', 'REB', 'SEASON']].head(10))
"""),
])

# ─────────────────────────────────────────────────────────────
# NOTEBOOK 02 — Cleaning & Normalization
# ─────────────────────────────────────────────────────────────
nb02 = nb([
    md("# 02 · Cleaning & Normalization\nBuilds before/after windows, applies VADER sentiment, and pace-adjusts stats across eras.\n\nOutputs: `data/processed/windows.csv` and `data/processed/normalized.csv`"),
    code("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_theme(style='whitegrid', palette='muted')
"""),
    code("""\
# Step 1: Score sentiment on lyric snippets
from src.sentiment import score_lyrics
df_sentiment = score_lyrics('../data/raw/mentions.csv')
print(df_sentiment[['player','artist','lyric','vader_compound']].head(10))
"""),
    code("""\
# Visualize sentiment distribution by mention type
fig, ax = plt.subplots(figsize=(10, 5))
for t, color in [('compliment','#1A8C50'), ('neutral','#888'), ('diss','#C42B2B')]:
    sub = df_sentiment[df_sentiment['mention_type'] == t]['vader_compound']
    ax.hist(sub, bins=20, alpha=0.6, label=t, color=color)
ax.axvline(0, color='black', linestyle='--', alpha=0.4)
ax.set_xlabel('VADER Compound Sentiment Score')
ax.set_ylabel('Count')
ax.set_title('Lyric Sentiment Distribution by Mention Type')
ax.legend()
plt.tight_layout()
plt.savefig('../data/processed/sentiment_distribution.png', dpi=150)
plt.show()
print("Note: sentiment scores don't always match manual labels — that's the point.")
"""),
    code("""\
# Step 2: Build before/after windows from game logs
from src.build_windows import build_windows
df_windows = build_windows()
print(f"Windows shape: {df_windows.shape}")
print(df_windows[['player','mention_type','before_PTS_mean','after_30g_PTS_mean']].head(10))
"""),
    code("""\
# Check for missing data
missing = df_windows.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
print(f"\\nComplete rows: {df_windows.dropna(subset=['after_30g_PTS_mean']).shape[0]}")
"""),
    code("""\
# Step 3: Pace-adjust and compute deltas
from src.normalize import normalize
df_norm = normalize()
print("\\nDelta columns created:")
delta_cols = [c for c in df_norm.columns if c.startswith('delta_')]
print(delta_cols)
"""),
    code("""\
# Preview key delta stats
print(df_norm[['player','mention_type','delta_after_30g_PTS_adj',
               'delta_after_30g_AST_adj','delta_after_30g_REB_adj',
               'composite_delta_30g']].dropna().head(15))
"""),
    code("""\
# Distribution of composite deltas
fig, ax = plt.subplots(figsize=(10, 5))
df_plot = df_norm.dropna(subset=['composite_delta_30g'])
ax.hist(df_plot['composite_delta_30g'], bins=25, color='#E05500', alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='No change')
ax.axvline(df_plot['composite_delta_30g'].mean(), color='#C48A00', linestyle='-',
           linewidth=2, label=f"Mean = {df_plot['composite_delta_30g'].mean():.2f}")
ax.set_xlabel('Composite Performance Delta (pace-adjusted, 30-game window)')
ax.set_ylabel('Number of Mentions')
ax.set_title('Distribution of Post-Mention Performance Change')
ax.legend()
plt.tight_layout()
plt.savefig('../data/processed/delta_distribution.png', dpi=150)
plt.show()
"""),
])

# ─────────────────────────────────────────────────────────────
# NOTEBOOK 03 — EDA
# ─────────────────────────────────────────────────────────────
nb03 = nb([
    md("# 03 · Exploratory Data Analysis\nDistributions, outliers, breakdowns by era, position, type, and artist tier."),
    code("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_theme(style='whitegrid', palette='muted')
df = pd.read_csv('../data/processed/normalized.csv')
df_valid = df.dropna(subset=['composite_delta_30g'])
print(f"Analyzing {len(df_valid)} complete mentions")
"""),
    code("""\
# 1. Breakdown by mention type
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stats = ['PTS', 'AST', 'REB']
for ax, stat in zip(axes, stats):
    col = f'delta_after_30g_{stat}_adj'
    data = [df_valid[df_valid['mention_type']==t][col].dropna() for t in ['compliment','neutral','diss']]
    ax.boxplot(data, labels=['Compliment', 'Neutral', 'Diss'],
               patch_artist=True,
               boxprops=dict(facecolor='#F7F4EE'),
               medianprops=dict(color='#E05500', linewidth=2))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Δ{stat} by Mention Type')
    ax.set_ylabel(f'Pace-adjusted Δ{stat}')
plt.suptitle('30-Game Post-Mention Performance Delta by Type', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('../data/processed/eda_by_type.png', dpi=150)
plt.show()
"""),
    code("""\
# 2. By era
fig, ax = plt.subplots(figsize=(10, 6))
era_means = df_valid.groupby('era')['composite_delta_30g'].mean().sort_index()
colors = ['#C84E00' if v > 0 else '#2B4FC4' for v in era_means.values]
bars = ax.bar(era_means.index, era_means.values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax.axhline(0, color='black', linestyle='--', alpha=0.4)
ax.set_title('Average Composite Performance Delta by Era')
ax.set_ylabel('Mean Composite Δ (pace-adjusted)')
for bar, val in zip(bars, era_means.values):
    ax.text(bar.get_x()+bar.get_width()/2, val + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/eda_by_era.png', dpi=150)
plt.show()
"""),
    code("""\
# 3. Artist tier vs composite delta (scatter)
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'compliment': '#1A8C50', 'neutral': '#888', 'diss': '#C42B2B'}
for t in ['compliment', 'neutral', 'diss']:
    sub = df_valid[df_valid['mention_type'] == t]
    ax.scatter(sub['artist_tier'], sub['composite_delta_30g'],
               label=t, color=colors[t], alpha=0.65, s=70, edgecolors='white', linewidths=0.5)
# Trend line
z = np.polyfit(df_valid['artist_tier'].dropna(), df_valid['composite_delta_30g'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(1, 3, 100)
ax.plot(x_line, p(x_line), 'k--', alpha=0.4, linewidth=1.5, label='Overall trend')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Artist Tier Multiplier')
ax.set_ylabel('Composite Δ Performance (30g, pace-adj.)')
ax.set_title('Does Artist Reach Predict Performance Change?')
ax.legend()
plt.tight_layout()
plt.savefig('../data/processed/eda_tier_scatter.png', dpi=150)
plt.show()
"""),
    code("""\
# 4. Sentiment vs delta (scatter) — the key plot
if 'vader_compound' in df_valid.columns:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, stat in zip(axes, ['PTS', 'AST', 'REB']):
        col = f'delta_after_30g_{stat}_adj'
        sub = df_valid[['vader_compound', col, 'mention_type']].dropna()
        for t in ['compliment', 'neutral', 'diss']:
            s = sub[sub['mention_type'] == t]
            ax.scatter(s['vader_compound'], s[col], label=t,
                       color=colors[t], alpha=0.6, s=60, edgecolors='white', linewidths=0.4)
        z = np.polyfit(sub['vader_compound'], sub[col], 1)
        x_line = np.linspace(-1, 1, 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'k--', alpha=0.4, linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('VADER Sentiment (-1 to +1)')
        ax.set_ylabel(f'Δ{stat}')
        ax.set_title(f'Sentiment vs Δ{stat}')
    axes[0].legend()
    plt.suptitle('Lyric Sentiment vs Post-Mention Performance Change', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('../data/processed/eda_sentiment_scatter.png', dpi=150)
    plt.show()
"""),
    code("""\
# 5. Top 15 biggest impacts (both directions)
top = df_valid.nlargest(8, 'composite_delta_30g')
bot = df_valid.nsmallest(7, 'composite_delta_30g')
top_df = pd.concat([top, bot]).sort_values('composite_delta_30g', ascending=True)

fig, ax = plt.subplots(figsize=(10, 9))
colors_bar = ['#1A8C50' if v > 0 else '#C42B2B' for v in top_df['composite_delta_30g']]
labels = [f"{r['player']}\\n({r['artist']}, {r['era']})" for _, r in top_df.iterrows()]
ax.barh(range(len(top_df)), top_df['composite_delta_30g'], color=colors_bar, alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(top_df)))
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Composite Performance Delta (pace-adjusted, 30g)')
ax.set_title('Biggest Performance Shifts After a Hip-Hop Mention')
plt.tight_layout()
plt.savefig('../data/processed/eda_top_impacts.png', dpi=150)
plt.show()
"""),
])

# ─────────────────────────────────────────────────────────────
# NOTEBOOK 04 — Statistical Analysis
# ─────────────────────────────────────────────────────────────
nb04 = nb([
    md("# 04 · Statistical Analysis\nPaired t-tests, Mann-Whitney U, Pearson correlations, and OLS regression.\n\n**Research Questions:**\n1. Does a hip-hop mention significantly change performance?\n2. Does mention type (compliment/diss) predict the direction?\n3. Does artist tier correlate with magnitude of change?\n4. Does VADER sentiment predict outcome better than manual type?"),
    code("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import pingouin as pg

%matplotlib inline
sns.set_theme(style='whitegrid')
df = pd.read_csv('../data/processed/normalized.csv')
df_valid = df.dropna(subset=['composite_delta_30g', 'before_PTS_mean', 'after_30g_PTS_mean'])
print(f"N = {len(df_valid)} complete mention windows")
"""),
    code("""\
# ── Q1: Paired t-tests — does a mention change performance? ──
print("=" * 60)
print("Q1: PAIRED T-TESTS — Before vs After (30-game window)")
print("=" * 60)
for stat in ['PTS', 'AST', 'REB']:
    before = df_valid[f'before_{stat}_mean']
    after  = df_valid[f'after_30g_{stat}_mean']
    t, p = stats.ttest_rel(after.dropna(), before.dropna())
    delta_mean = (after - before).mean()
    d = (after - before).mean() / (after - before).std()
    sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
    print(f"  {stat}: Δ={delta_mean:+.3f}  t={t:.3f}  p={p:.4f}  Cohen's d={d:.3f}  {sig}")
"""),
    code("""\
# ── Q2: Do compliments vs disses differ? (Mann-Whitney U) ──
print("\\n" + "=" * 60)
print("Q2: MANN-WHITNEY U — Compliment vs Diss")
print("=" * 60)
for stat in ['PTS', 'AST', 'REB']:
    col = f'delta_after_30g_{stat}_adj'
    comp = df_valid[df_valid['mention_type']=='compliment'][col].dropna()
    diss = df_valid[df_valid['mention_type']=='diss'][col].dropna()
    if len(comp) >= 3 and len(diss) >= 3:
        u, p = stats.mannwhitneyu(comp, diss, alternative='two-sided')
        sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
        print(f"  {stat}: Compliment mean={comp.mean():+.3f}  Diss mean={diss.mean():+.3f}")
        print(f"         U={u:.1f}  p={p:.4f}  {sig}")
"""),
    code("""\
# ── Q3: Artist tier correlation ──
print("\\n" + "=" * 60)
print("Q3: PEARSON CORRELATION — Artist Tier vs Stat Delta")
print("=" * 60)
for stat in ['PTS', 'AST', 'REB']:
    col = f'delta_after_30g_{stat}_adj'
    valid = df_valid[['artist_tier', col]].dropna()
    r, p = stats.pearsonr(valid['artist_tier'], valid[col])
    sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
    print(f"  Artist Tier → Δ{stat}: r={r:.3f}  p={p:.4f}  {sig}")
"""),
    code("""\
# ── Q4: VADER sentiment vs delta ──
if 'vader_compound' in df_valid.columns:
    print("\\n" + "=" * 60)
    print("Q4: PEARSON CORRELATION — VADER Sentiment vs Stat Delta")
    print("=" * 60)
    for stat in ['PTS', 'AST', 'REB']:
        col = f'delta_after_30g_{stat}_adj'
        valid = df_valid[['vader_compound', col]].dropna()
        r, p = stats.pearsonr(valid['vader_compound'], valid[col])
        sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
        print(f"  VADER → Δ{stat}: r={r:.3f}  p={p:.4f}  {sig}")
"""),
    code("""\
# ── Correlation Heatmap ──
corr_cols = ['artist_tier', 'composite_delta_30g',
             'delta_after_30g_PTS_adj', 'delta_after_30g_AST_adj', 'delta_after_30g_REB_adj']
if 'vader_compound' in df_valid.columns:
    corr_cols.insert(1, 'vader_compound')
corr_df = df_valid[corr_cols].dropna()
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix — Key Variables', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('../data/processed/correlation_heatmap.png', dpi=150)
plt.show()
"""),
    code("""\
# ── OLS Regression ──
print("\\n" + "=" * 60)
print("MULTIPLE REGRESSION: composite_delta ~ predictors")
print("=" * 60)

# Encode mention_type as dummy vars
df_reg = df_valid.copy()
df_reg['type_compliment'] = (df_reg['mention_type'] == 'compliment').astype(int)
df_reg['type_diss']       = (df_reg['mention_type'] == 'diss').astype(int)

formula = 'composite_delta_30g ~ artist_tier + type_compliment + type_diss'
if 'vader_compound' in df_reg.columns:
    formula += ' + vader_compound'

model = smf.ols(formula=formula, data=df_reg.dropna()).fit()
print(model.summary())
"""),
    code("""\
# Coefficient plot
fig, ax = plt.subplots(figsize=(9, 5))
coefs = model.params.drop('Intercept')
cis   = model.conf_int().drop('Intercept')
colors_coef = ['#1A8C50' if v > 0 else '#C42B2B' for v in coefs]
ax.barh(coefs.index, coefs.values, color=colors_coef, alpha=0.75, edgecolor='white')
ax.errorbar(coefs.values, range(len(coefs)),
            xerr=[coefs.values - cis[0].values, cis[1].values - coefs.values],
            fmt='none', color='#333', linewidth=1.5, capsize=4)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Regression Coefficients (95% CI)\nDependent Variable: Composite Performance Delta')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('../data/processed/regression_coefficients.png', dpi=150)
plt.show()

print(f"\\nModel R² = {model.rsquared:.3f}  (Adjusted R² = {model.rsquared_adj:.3f})")
print(f"F-statistic p-value = {model.f_pvalue:.4f}")
"""),
])

# ─────────────────────────────────────────────────────────────
# NOTEBOOK 05 — Export
# ─────────────────────────────────────────────────────────────
nb05 = nb([
    md("# 05 · Export\nGenerates `data/processed/results.json` — the single file consumed by `index.html`.\n\nRun this after all previous notebooks are complete."),
    code("""\
import sys
sys.path.insert(0, '..')
from src.export import build_export
import json
from pathlib import Path

results = build_export()
print("\\n── KEY FINDINGS ──────────────────────────────────────")
for k, v in results['key_findings'].items():
    print(f"  {k}: {v}")
"""),
    code("""\
# Verify the JSON is valid and readable
with open('../data/processed/results.json') as f:
    check = json.load(f)
print(f"✓ results.json is valid JSON")
print(f"  Mentions: {len(check['mentions'])}")
print(f"  Top impacts: {len(check['top_impacts'])}")
print(f"  Generated at: {check['generated_at']}")
"""),
    code("""\
# Preview top 5 impacts
print("\\n── TOP 5 IMPACTS ─────────────────────────────────────")
for m in check['top_impacts'][:5]:
    d = m['deltas']
    print(f"  {m['player']} × {m['artist']} ({m['era']}, {m['mention_type']})")
    print(f"    ΔPTS={d['pts_30g']:+}  ΔAST={d['ast_30g']:+}  ΔREB={d['reb_30g']:+}  Composite={d['composite_30g']:+}")
    print()
"""),
    code("""\
# Check file size — results.json should be < 2MB for GitHub Pages
import os
size_kb = os.path.getsize('../data/processed/results.json') / 1024
print(f"results.json size: {size_kb:.1f} KB")
if size_kb > 1000:
    print("⚠ File is large — consider minifying before committing")
else:
    print("✓ Size is fine for GitHub Pages")
print("\\n✓ All done! Open index.html in your browser to see the dashboard.")
"""),
])

# ── WRITE NOTEBOOKS ──────────────────────────────────────────
notebooks = [
    ("01_data_collection.ipynb", nb01),
    ("02_cleaning.ipynb", nb02),
    ("03_eda.ipynb", nb03),
    ("04_statistics.ipynb", nb04),
    ("05_export.ipynb", nb05),
]

for fname, notebook in notebooks:
    path = NB_DIR / fname
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print(f"✓ Written: {path}")

print("\nAll notebooks generated.")
