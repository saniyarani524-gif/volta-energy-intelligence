# VOLTA — Energy Market Intelligence & Battery Trading Desk

Turn Spanish day-ahead prices into CHARGE / DISCHARGE / HOLD cards — and prove a
€70m battery still does not pay back.

**Clone → `pip install -r requirements.txt` → `streamlit run app.py`.**
The tidy 2015–18 tape and 2018 caches are in this repo. You do not need Kaggle
to open the desk.

| 2018 test (temporal split) | Number |
|---|---|
| Hours in the tape | 35,064 (2015-01-01 → 2018-12-31, Europe/Madrid) |
| TSO day-ahead price MAE | **€8.86** |
| VOLTA XGBoost MAE | **€3.69** (−58% vs TSO) |
| Walk-forward MAE | €3.49 |
| 2016–18 VOLTA settled P&L | **€1.64m** (~80% of perfect foresight) |
| Perfect-foresight 2015–18 | €3.13m (~€784k / year) |
| Payback / NPV @ €350/kWh, 7% WACC | **89 years · −€62.9m** |
| 2018 actions | 669 CHARGE · 651 DISCHARGE · 7,440 HOLD |
| Days sat out | 52 (spread < round-trip + €3) |

The last two rows are the product: a desk that holds, not a Kaggle loop that always trades.

## Run the desk

```bash
git clone https://github.com/saniyarani524-gif/volta-energy-intelligence.git
cd volta-energy-intelligence
python3 -m pip install -r requirements.txt
python3 -m src.doctor
python3 -m streamlit run app.py
```

Doctor must print `read market: (35064, 28)`. Then open http://localhost:8501.

## Pages (left sidebar)

| Group | Page | What it proves |
|---|---|---|
| Desk | **Command** | Selected day, top cards, € expected |
| Desk | **Cards** | CHARGE / DISCHARGE / HOLD · why · confidence |
| Desk | **Dispatch** | Daily LP · SOC · VOLTA vs TSO vs perfect |
| Market | **Pulse** | 4-year spot, load, hour × month |
| Market | **Stack** | Duck curve, merit order, Jan-2017 spikes |
| Market | **Forecast** | Beat the TSO · P10–P90 band |
| Results | **Ledger** | Settled P&L · **does not pay back** |
| Results | **Risk** | Day-level scale · 52 days sat out |
| Results | **Notes** | Assumptions, leakage contract, config |

## Pipeline (audit trail)

Each notebook writes artifacts the next notebook and the desk consume.

1. `notebooks/01_eda.ipynb` — tidy 35,064 × 28 parquet, duck curve, merit order, TSO gap
2. `notebooks/02_forecasting.ipynb` — leakage-safe 24h-ahead XGB vs TSO vs naive-24h
3. `notebooks/03_dispatch.ipynb` — 100 MW / 200 MWh daily LP, settle on realised spot
4. `notebooks/04_decisions.ipynb` — cards with why, day-level risk scale, flags

```
pytest -q          # 24 tests: features, LP invariants, cards, loaders
python -m src.doctor
```

Re-running 01–04 from raw CSVs is optional. Drop the Kaggle files into
`data/raw/` (see `data/raw/README.md`). Raw CSVs are **not** in git (~25 MB).

## What is in git vs not

| In the repo | Not in the repo |
|---|---|
| `data/processed/market.parquet` (2.7 MB) | `data/raw/*.csv` |
| `cache/*.parquet` (2016–18 forecasts, dispatch, 2018 cards) | `models/*.json` (rebuild from nb 02) |
| notebooks 01–04 + `notebooks/figures/` | secrets / `kaggle.json` |

## Rules we did not break

- Temporal splits only. Never shuffle a time series.
- Realised price / load / generation may only enter as a lag ≥ 24 h.
- Day-level risk scale only (hour-level MW edits break SOC).
- SPIKE is a flag, not a silent rewrite.
- Every number in this README regenerates from notebooks 01→04.

## Stack

Python · pandas · XGBoost · scipy.linprog · Plotly · Streamlit · pytest

Data: [Hourly energy demand, generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (Spain ENTSO-E / REE + 5-city weather, CC0).

---

Built by **Saniya** · public market data (CC0) · the battery does not pay back, and we say so.
