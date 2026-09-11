"""Loaders for the Streamlit desk.

Looks in every plausible repo root (cwd, this file, VOLTA_ROOT) so
``streamlit run`` from the wrong folder still finds the tape. Rebuilds
from the Kaggle CSVs if the parquet is missing.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import PRICE_FEATURES
from src.market_io import (
    build_and_save_market,
    energy_csv,
    find_file,
    market_parquet,
    weather_csv,
)

READ_ERRORS: dict[str, str] = {}

DEFAULT_DAY = date(2018, 1, 25)

PRESET_DAYS = [
    (date(2018, 1, 25), "Squeeze", "Fat evening print, engine wants the peak."),
    (date(2018, 6, 21), "Summer solstice", "Solar noon vs evening ramp."),
    (date(2018, 3, 4), "Spring", "A quieter weekday — most hours HOLD."),
    (date(2017, 1, 18), "2017 spike week", "Perfect-foresight ceiling day."),
    (date(2016, 2, 14), "Wet 2016", "Hydro-cheap year, thinner spreads."),
]


def _read(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except ImportError as e:
        READ_ERRORS[str(path)] = (
            f"{e}. Install pyarrow in the SAME venv as streamlit:  pip install pyarrow"
        )
        return None
    except Exception as e:
        READ_ERRORS[str(path)] = f"{type(e).__name__}: {e}"
        return None


def read_market() -> pd.DataFrame | None:
    return _read(market_parquet())


def read_forecasts_2018() -> pd.DataFrame | None:
    return _read(find_file("cache", "test_2018_forecasts.parquet"))


def read_walkforward() -> pd.DataFrame | None:
    return _read(find_file("cache", "wf_forecasts_2016_2018.parquet"))


def read_cards() -> pd.DataFrame | None:
    return _read(find_file("cache", "action_cards_2018.parquet"))


def read_dispatch_daily() -> pd.DataFrame | None:
    parts = []
    for name in ("perfect", "volta", "tso"):
        df = _read(find_file("cache", f"dispatch_{name}_daily.parquet"))
        if df is None:
            continue
        if "policy" not in df.columns:
            df = df.copy()
            df["policy"] = name
        parts.append(df)
    if not parts:
        return None
    return pd.concat(parts).sort_index()


def read_dispatch_hourly_perfect() -> pd.DataFrame | None:
    return _read(find_file("cache", "dispatch_hourly_perfect.parquet"))


def data_status() -> dict[str, bool]:
    return {
        "market": market_parquet() is not None,
        "forecasts_2018": find_file("cache", "test_2018_forecasts.parquet") is not None,
        "walkforward": find_file("cache", "wf_forecasts_2016_2018.parquet") is not None,
        "cards": find_file("cache", "action_cards_2018.parquet") is not None,
        "dispatch": find_file("cache", "dispatch_volta_daily.parquet") is not None,
        "models": find_file("models", "price_xgb.json") is not None,
        "raw_csvs": energy_csv() is not None and weather_csv() is not None,
    }


def slice_day(df: pd.DataFrame, day: date) -> pd.DataFrame:
    key = pd.Timestamp(day).strftime("%Y-%m-%d")
    try:
        out = df.loc[key]
    except KeyError:
        return df.iloc[0:0]
    if isinstance(out, pd.Series):
        out = out.to_frame().T
    return out


def available_days(df: pd.DataFrame | None) -> tuple[date, date] | None:
    if df is None or df.empty:
        return None
    lo = df.index.min()
    hi = df.index.max()
    return lo.date(), hi.date()


def clamp_day(day: date, span: tuple[date, date] | None) -> date:
    if span is None:
        return day
    lo, hi = span
    return min(max(day, lo), hi)


def mae_table_2018(fc: pd.DataFrame) -> pd.DataFrame:
    y = fc["price_actual"]
    naive = y.shift(24)
    rows = []

    def row(name, pred):
        err = (pred - y).abs()
        mape = float(np.mean(err / np.clip(y.abs(), 1e-6, None)) * 100)
        rows.append(
            {
                "model": name,
                "MAE": float(err.mean()),
                "RMSE": float(np.sqrt(((pred - y) ** 2).mean())),
                "MAPE": mape,
            }
        )

    row("TSO day-ahead", fc["price_da"])
    row("Naive-24h", naive)
    if "price_pred" in fc.columns:
        row("VOLTA XGB", fc["price_pred"])
    if "price_p50" in fc.columns:
        row("VOLTA P50", fc["price_p50"])
    if "price_pred_wf" in fc.columns:
        row("VOLTA walk-forward", fc["price_pred_wf"])
    return pd.DataFrame(rows).set_index("model")


def generation_mix(market: pd.DataFrame) -> pd.DataFrame:
    mix = pd.DataFrame(
        {
            "nuclear": market["gen_nuclear"],
            "wind": market["gen_wind"],
            "hydro": market["gen_hydro"],
            "gas": market["gen_gas"],
            "coal": market["gen_coal"] + market["gen_lignite"],
            "solar": market["gen_solar"],
            "other": (
                market["gen_biomass"]
                + market["gen_oil"]
                + market["gen_waste"]
                + market["gen_other"]
                + market["gen_other_re"]
            ),
        },
        index=market.index,
    )
    return mix


def feature_importance() -> pd.Series | None:
    path = find_file("models", "price_xgb.json")
    if path is None:
        return None
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    model = XGBRegressor()
    model.load_model(path)
    n = len(model.feature_importances_)
    names = list(PRICE_FEATURES)[:n]
    return pd.Series(model.feature_importances_, index=names).sort_values()


def next_weekday(day: date, delta: int) -> date:
    return day + timedelta(days=delta)


def try_build_market() -> Path:
    return build_and_save_market()
