# VOLTA — Energy Market Intelligence & Battery Trading Desk

> Turn market prices, weather and renewables into smarter trading decisions.

Portfolio project: **real** electricity market data (Spain, ENTSO-E / Red Eléctrica, 2015–2018,
via Kaggle) → day-ahead price forecasting benchmarked against the grid operator's own forecast →
battery arbitrage optimizer (100 MW / 200 MWh) → explainable trading decisions → 9-page Streamlit
control-room UI.

**Status:** Phase 5 — 9-page desk. Numbers come from notebooks 01–04. The battery does **not** pay
back on 2015–18 Spanish spreads; that sentence is the product.

## Live app

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Needs `data/processed/market.parquet` (notebook 01). Forecast / dispatch / card pages light up
after notebooks 02–04. The desk never invents a number.

| Page | What it is |
|---|---|
| Command | Replay desk · selected day · top cards |
| Cards | CHARGE / DISCHARGE / HOLD with why, confidence, flags |
| Dispatch | Daily LP · VOLTA P50 vs TSO vs perfect foresight |
| Pulse | Four years of spot, load, hour × month heatmap |
| Stack | Duck curve, merit order, spike anatomy, generation mix |
| Forecast | Beat the TSO on price · quantile band · leakage-safe features |
| Ledger | Settled P&L · capex / NPV / payback (honest) |
| Risk | Day-level scale · flags · live what-if battery |
| Notes | Assumptions, leakage contract, config, how to reproduce |

## Data

| File | Rows | Contents |
|---|---|---|
| `data/raw/energy_dataset.csv` | 35,064 × 29 | hourly price (€/MWh), load, TSO forecasts, generation by 20+ sources (2015–2018) |
| `data/raw/weather_features.csv` | 178,396 × 17 | hourly weather for Madrid, Barcelona, Valencia, Seville, Bilbao |

Source: [Kaggle — Hourly energy demand, generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (CC0).
Data files are git-ignored — they stay local.

## Structure

```
app.py                 Streamlit entry · 9-page navigation
src/ui.py              design system (teal / amber on near-black)
src/app_data.py        parquet loaders — return None if a notebook has not been run
src/views/             one render() per page
src/config.py          battery, splits, decision thresholds
src/features.py        leakage-safe 24h-ahead feature contract
src/models.py          MAE / XGB / walk-forward
src/optimizer.py       daily LP + settle + economics
src/decisions.py       CHARGE / DISCHARGE / HOLD cards
notebooks/             01 EDA · 02 forecast · 03 dispatch · 04 decisions
tests/                 unit + smoke
```

## Roadmap

1. **Foundations** — tidy hourly frame, duck curve, merit order, spike taxonomy
2. **Forecasting** — XGBoost day-ahead price vs TSO, quantile bands, walk-forward
3. **Optimization** — LP dispatch, 3-year € P&L, degradation, project finance
4. **Decisions** — explainable cards, day-level risk scale, flags
5. **Product** — this 9-page desk

---
Built by **Saniya** · every number from public real market data (CC0).
