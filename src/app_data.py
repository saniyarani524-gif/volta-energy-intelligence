"""Loaders for the Streamlit desk.

Cache / processed parquet is produced by notebooks 01–04. Nothing here
invents a number — if the file is missing we return None and the page
asks the user to run the notebook.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE, MARKET_PARQUET, MODELS
from src.features import PRICE_FEATURES

DEFAULT_DAY = date(2018, 1, 25)  # squeeze cousin of the 2017 spike cluster

PRESET_DAYS = [
    (date(2018, 1, 25), "Squeeze", "Fat evening print, engine wants the peak."),
    (date(2018, 6, 21), "Summer solstice", "Solar noon vs evening ramp."),
    (date(2018, 3, 4), "Spring", "A quieter weekday — most hours HOLD."),
    (date(2017, 1, 18), "2017 spike week", "Perfect-foresight ceiling day."),
    (date(2016, 2, 14), "Wet 2016", "Hydro-cheap year, thinner spreads."),
]


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def read_market() -> pd.DataFrame | None:
    return _read(MARKET_PARQUET)


def read_forecasts_2018() -> pd.DataFrame | None:
    return _read(CACHE / "test_2018_forecasts.parquet")


def read_walkforward() -> pd.DataFrame | None:
    return _read(CACHE / "wf_forecasts_2016_2018.parquet")


def read_cards() -> pd.DataFrame | None:
    return _read(CACHE / "action_cards_2018.parquet")


def read_dispatch_daily() -> pd.DataFrame | None:
    parts = []
    for name in ("perfect", "volta", "tso"):
        df = _read(CACHE / f"dispatch_{name}_daily.parquet")
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
    return _read(CACHE / "dispatch_hourly_perfect.parquet")


def data_status() -> dict[str, bool]:
    return {
        "market": MARKET_PARQUET.exists(),
        "forecasts_2018": (CACHE / "test_2018_forecasts.parquet").exists(),
        "walkforward": (CACHE / "wf_forecasts_2016_2018.parquet").exists(),
        "cards": (CACHE / "action_cards_2018.parquet").exists(),
        "dispatch": (CACHE / "dispatch_volta_daily.parquet").exists(),
        "models": (MODELS / "price_xgb.json").exists(),
    }


def slice_day(df: pd.DataFrame, day: date) -> pd.DataFrame:
    """Timezone-safe calendar-day slice."""
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
    path = MODELS / "price_xgb.json"
    if not path.exists():
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
