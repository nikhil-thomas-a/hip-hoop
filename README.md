# HIP HOOP 🎤🏀
https://nikhil-thomas-a.github.io/hip-hoop/

> *Does a hip-hop mention change how an NBA player performs?*

A full end-to-end data science project analyzing the statistical relationship between hip-hop cultural mentions and NBA player performance. Real game logs. Real lyrics. Real math.

---

## Project Structure

```
hip-hoop/
├── data/
│   ├── raw/
│   │   ├── mentions.csv          # 110 hip-hop NBA mentions with release dates
│   │   └── game_logs/            # per-player game log CSVs (generated, gitignored)
│   └── processed/
│       ├── mentions_with_sentiment.csv
│       ├── windows.csv           # before/after stat windows per mention
│       ├── normalized.csv        # pace-adjusted, era-normalized
│       └── results.json          # ← final output read by index.html
├── notebooks/
│   ├── 01_data_collection.ipynb  # NBA API + Genius API pulls
│   ├── 02_cleaning.ipynb         # sentiment, windows, normalization
│   ├── 03_eda.ipynb              # exploratory analysis + charts
│   ├── 04_statistics.ipynb       # t-tests, correlations, regression
│   └── 05_export.ipynb           # generates results.json
├── src/
│   ├── fetch_gamelogs.py         # nba_api wrapper
│   ├── fetch_lyrics.py           # Genius API fetcher
│   ├── sentiment.py              # VADER sentiment scoring
│   ├── build_windows.py          # before/after window construction
│   ├── normalize.py              # pace + era normalization
│   └── export.py                 # builds results.json
├── index.html                    # interactive executive summary
├── generate_notebooks.py         # regenerates all notebooks from source
├── requirements.txt
├── .env.example                  # copy to .env and add your API key
└── .gitignore
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/hip-hoop.git
cd hip-hoop
pip install -r requirements.txt
```

### 2. Add your Genius API key

```bash
cp .env.example .env
# Edit .env and add your GENIUS_API_KEY
# Get a free key at: https://genius.com/api-clients
```

> ⚠️ **Never commit your `.env` file.** It is in `.gitignore`.

### 3. Run the notebooks in order

```bash
jupyter notebook
```

Open and run each notebook in sequence:

| Notebook | What it does | Runtime |
|---|---|---|
| `01_data_collection` | Fetches NBA game logs + Genius metadata | ~20 min |
| `02_cleaning` | Sentiment scoring, windows, normalization | ~2 min |
| `03_eda` | Exploratory charts and distributions | ~1 min |
| `04_statistics` | t-tests, correlations, regression | ~1 min |
| `05_export` | Generates `data/processed/results.json` | ~30 sec |

### 4. Open the dashboard

```bash
# Option A: open directly in browser
open index.html

# Option B: serve locally (avoids fetch() CORS issues)
python3 -m http.server 8080
# then visit http://localhost:8080
```

---

## Methodology

### Data Sources
- **NBA game logs** — `nba_api` library, pulling per-game stats (PTS, AST, REB) for each player
- **Release dates** — from `mentions.csv`, cross-referenced with Genius API
- **Sentiment** — VADER applied to each lyric snippet

### Statistical Windows
For each mention, we construct 5 windows relative to the song release date:
- **Baseline**: 30 games before
- **After 1g**: next 1 game
- **After 10g**: next ~10 games (~2 weeks)
- **After 30g**: next ~30 games (~1 month)
- **After season**: remainder of that NBA season

### Normalization
- **Pace adjustment**: scales stats to a 2010s reference pace (NBA pace has varied from ~89 to ~100 possessions/game across eras)
- **Era z-scores**: standardizes deltas within each era group

### Statistical Tests
- **Paired t-tests**: before vs. after for PTS, AST, REB
- **Mann-Whitney U**: compliment vs. diss group comparison
- **Pearson correlation**: artist tier and VADER score vs. performance delta
- **OLS regression**: composite delta ~ artist tier + mention type + sentiment

### Composite Impact Score
`PTS delta (40%) + AST delta (30%) + REB delta (30%)` — pace-adjusted, 30-game window.

---

## Key Research Questions

1. Does a hip-hop mention significantly change performance? *(paired t-test)*
2. Does mention type (compliment vs. diss) predict direction? *(Mann-Whitney U)*
3. Does artist tier correlate with magnitude of change? *(Pearson r)*
4. Does continuous sentiment score outperform binary classification? *(regression)*

---

## Dataset: 110 Mentions

Spanning 1992–2022, covering players from Shaq and MJ in the 1990s to Luka, Ja Morant, and Anthony Edwards in the 2020s. Artists from Notorious B.I.G., Jay-Z, and Nas through Drake, Kendrick Lamar, and Travis Scott.

---

## Limitations

This is a **correlational study**. We cannot claim causation — basketball performance is driven by hundreds of confounding factors (opponent strength, injury status, rest, home/away). We report p-values and effect sizes honestly and acknowledge when results are not statistically significant. The dataset of ~110 mentions is appropriate for exploratory analysis but underpowered for strong causal claims.

**Selection bias**: we only captured famous, well-documented mentions. Unknown references are underrepresented.

---

## Live Demo

Deployed via GitHub Pages: `https://yourusername.github.io/hip-hoop`

*Note: GitHub Pages serves static files. The `results.json` in `data/processed/` must be committed for the live dashboard to work.*

---

## License

MIT — fork it, remix it, cite it.

---

*Built for culture.*
