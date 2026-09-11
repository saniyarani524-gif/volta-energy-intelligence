"""VOLTA — global configuration & constants.

Real data: Spain hourly electricity market (ENTSO-E / Red Eléctrica via Kaggle),
31 Dec 2014 → 31 Dec 2018, plus weather for 5 cities.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW, PROCESSED = DATA / "raw", DATA / "processed"
CACHE, MODELS = ROOT / "cache", ROOT / "models"

ENERGY_CSV = RAW / "energy_dataset.csv"
WEATHER_CSV = RAW / "weather_features.csv"
MARKET_PARQUET = PROCESSED / "market.parquet"

# ---------------------------------------------------------------- project
COMPANY = "VOLTA TRADING DESK"
TAGLINE = "Turn market prices, weather and renewables into smarter trading decisions."
TZ = "Europe/Madrid"

# ---------------------------------------------------------------- battery asset
BATTERY_POWER_MW = 100.0          # max charge/discharge rate
BATTERY_CAPACITY_MWH = 200.0      # energy capacity (2h duration asset)
ROUND_TRIP_EFFICIENCY = 0.88      # charge_eff * discharge_eff (split equally: sqrt)
MIN_SOC_PCT = 0.10                # floor state of charge
MAX_CYCLES_PER_DAY = 1.0          # degradation / warranty constraint v1
DEGRADATION_COST_EUR_PER_MWH = 3.0  # € per MWh discharged to the grid (assumption)

# Economics — 2018-era turnkey Li-ion, not 2024 pack prices. Interview will ask.
BATTERY_CAPEX_EUR_PER_KWH = 350.0
WACC = 0.07
ASSET_LIFE_YEARS = 15

# ---------------------------------------------------------------- market
CITY_WEIGHTS = {  # population-weighted weather aggregation (approx, 5 cities)
    "Madrid": 0.28, "Barcelona": 0.27, "Valencia": 0.22,
    "Seville": 0.13, "Bilbao": 0.10,
}

# ---------------------------------------------------------------- forecasting
# Temporal split ONLY. Never shuffle a time series.
# Train: 2015–2017 inclusive. Test: full calendar 2018.
TRAIN_END = "2017-12-31 23:00"
TEST_START = "2018-01-01 00:00"
HORIZON_HOURS = 24          # day-ahead
MIN_LAG_HOURS = 24          # realised series may not be used closer than this

# XGBoost defaults (hist, CPU). Quantile models use a slightly shallower tree.
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "random_state": 7,
    "n_jobs": 4,
    "tree_method": "hist",
}
XGB_QUANTILE_PARAMS = {
    **XGB_PARAMS,
    "n_estimators": 300,
    "max_depth": 5,
}
QUANTILES = (0.1, 0.5, 0.9)

# ---------------------------------------------------------------- decisions (Phase 4)
ACTION_MW_EPS = 5.0          # below this, the hour is HOLD
UNCERTAINTY_SOFT = 0.25      # (P90-P10)/P50 → cut position in half
UNCERTAINTY_HARD = 0.45      # cut to 25%
MIN_POSITION_SCALE = 0.25
SPIKE_EUR = 90.0             # €/MWh — from Phase 1 spike anatomy
SPIKE_Z = 2.0                # vs the day's own forecast mean/std
