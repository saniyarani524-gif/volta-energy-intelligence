#  VOLTA — Energy Market Intelligence & Battery Trading Desk

> Turn market prices, weather and renewables into smarter trading decisions.

Portfolio project: **real** electricity market data (Spain, ENTSO-E / Red Eléctrica, 2015–2018,
via Kaggle) → day-ahead price forecasting benchmarked against the grid operator's own forecast →
battery arbitrage optimizer (100 MW / 200 MWh) → explainable trading decisions → Streamlit
control-room UI.

**Status:** 🚧 Week 1 of 12 — data acquisition ✓ · app shell live

## Live app
_Deployed on Streamlit Community Cloud — URL lands here after first deploy._

## Data

| File | Rows | Contents |
|---|---|---|
| `data/raw/energy_dataset.csv` | 35,064 × 29 | hourly price (€/MWh), load, TSO forecasts, generation by 20+ sources (2015–2018) |
| `data/raw/weather_features.csv` | 178,396 × 17 | hourly weather for Madrid, Barcelona, Valencia, Seville, Bilbao |

Source: [Kaggle — Hourly energy demand, generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (CC0).
Data files are git-ignored — they stay local.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt      # app deps + dev tools (kaggle, notebook)
streamlit run app.py
```

## Structure

```
app.py           Streamlit entrypoint
src/             config, pipeline, analytics, models, optimizer, pages (built weekly)
data/raw/        Kaggle CSVs (git-ignored)
notebooks/       EDA notebooks
tests/           smoke + unit tests
docs/            case study + figures (Week 12)
```

## Roadmap

1. **W1–3** Foundations — data pipeline, duck curve, merit order, spike taxonomy
2. **W4–6** Forecasting — XGBoost day-ahead price/load vs TSO benchmark, quantile bands
3. **W7–9** Optimization — LP dispatch, rolling horizon, 3-year € P&L backtest
4. **W10–12** Product — 9-page control-room UI, decision engine, case study, launch

---
Built by **Saniya** · every number from public real market data (CC0).
