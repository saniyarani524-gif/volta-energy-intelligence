"""Leakage-safe 24h-ahead feature builder.

Contract
--------
Anything realised (price_actual, load_actual, generation, net_load) may only
enter as a lag of >= 24 hours, or as a rolling statistic computed on a series
that has already been shifted by 24 hours.

Known at forecast time (legal, no shift):
- calendar (Madrid local)
- TSO day-ahead columns: price_da, load_da, forecast_solar_da, forecast_wind_da
- derived net_load_da = load_da - forecast_solar_da - forecast_wind_da
- weather at t, used as a stand-in for the D-1 NWP forecast (assumption)

This module is imported by the forecasting notebook AND by later app/pipeline
code so the feature set cannot silently drift.
"""
from __future__ import annotations

import pandas as pd

from src.config import MIN_LAG_HOURS

# National public holidays in Spain, 2015–2018 (not region-specific, not observed-on-Monday).
# Assumption: a Madrid-local calendar is close enough for a national price model.
SPAIN_HOLIDAYS = frozenset(
    pd.Timestamp(d).date()
    for d in [
        # 2015
        "2015-01-01", "2015-01-06", "2015-04-03", "2015-05-01", "2015-08-15",
        "2015-10-12", "2015-11-01", "2015-12-06", "2015-12-08", "2015-12-25",
        # 2016
        "2016-01-01", "2016-01-06", "2016-03-25", "2016-05-01", "2016-08-15",
        "2016-10-12", "2016-11-01", "2016-12-06", "2016-12-08", "2016-12-25",
        # 2017
        "2017-01-01", "2017-01-06", "2017-04-14", "2017-05-01", "2017-08-15",
        "2017-10-12", "2017-11-01", "2017-12-06", "2017-12-08", "2017-12-25",
        # 2018
        "2018-01-01", "2018-01-06", "2018-03-30", "2018-05-01", "2018-08-15",
        "2018-10-12", "2018-11-01", "2018-12-06", "2018-12-08", "2018-12-25",
    ]
)

# Same-hour realised columns. Must NEVER appear in a feature list.
REALISED_NOW = frozenset(
    {
        "price_actual",
        "load_actual",
        "net_load",
        "gen_biomass",
        "gen_lignite",
        "gen_gas",
        "gen_coal",
        "gen_oil",
        "gen_hydro_pump",
        "gen_hydro_ror",
        "gen_hydro_res",
        "gen_hydro",
        "gen_nuclear",
        "gen_other",
        "gen_other_re",
        "gen_solar",
        "gen_waste",
        "gen_wind",
    }
)

PRICE_FEATURES: tuple[str, ...] = (
    # calendar
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # TSO / OMIE day-ahead (known ~12:00 D-1)
    "price_da",
    "load_da",
    "forecast_solar_da",
    "forecast_wind_da",
    "net_load_da",
    # weather-as-NWP (assumption)
    "temp_c",
    "humidity",
    "wind_speed",
    "clouds",
    # realised lags >= 24h
    "price_lag24",
    "price_lag48",
    "price_lag72",
    "price_lag168",
    "load_lag24",
    "load_lag168",
    "wind_lag24",
    "solar_lag24",
    "gas_lag24",
    # rolling stats on the already-shifted price/load
    "price_roll24_mean",
    "price_roll168_mean",
    "price_roll24_std",
    "load_roll24_mean",
)

# Ablation: same model, but we refuse to use the operator's price number.
PRICE_FEATURES_NO_DA: tuple[str, ...] = tuple(c for c in PRICE_FEATURES if c != "price_da")

LOAD_FEATURES: tuple[str, ...] = (
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "load_da",
    "forecast_solar_da",
    "forecast_wind_da",
    "temp_c",
    "humidity",
    "wind_speed",
    "load_lag24",
    "load_lag48",
    "load_lag72",
    "load_lag168",
    "load_roll24_mean",
    "load_roll168_mean",
)


def _cyclic(frame: pd.DataFrame, col: str, period: int) -> None:
    import numpy as np

    frame[f"{col}_sin"] = np.sin(2 * np.pi * frame[col] / period)
    frame[f"{col}_cos"] = np.cos(2 * np.pi * frame[col] / period)


def build_feature_frame(market: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, TSO-derived, lag and rolling columns.

    Returns a copy. Does not dropna — the caller decides (train/test after lags
    need ~7 days of burn-in).
    """
    if MIN_LAG_HOURS != 24:
        raise ValueError("this builder hard-codes 24h as the minimum realised lag")

    df = market.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = [d in SPAIN_HOLIDAYS for d in idx.date]
    df["is_holiday"] = df["is_holiday"].astype(int)
    _cyclic(df, "hour", 24)
    _cyclic(df, "dow", 7)
    _cyclic(df, "month", 12)

    df["net_load_da"] = df["load_da"] - df["forecast_solar_da"] - df["forecast_wind_da"]

    for lag in (24, 48, 72, 168):
        df[f"price_lag{lag}"] = df["price_actual"].shift(lag)
        df[f"load_lag{lag}"] = df["load_actual"].shift(lag)
    df["wind_lag24"] = df["gen_wind"].shift(24)
    df["solar_lag24"] = df["gen_solar"].shift(24)
    df["gas_lag24"] = df["gen_gas"].shift(24)

    # Shift FIRST, then roll — otherwise the window includes hours we would not
    # yet have seen at 24h-ahead forecast time.
    price_s = df["price_actual"].shift(24)
    load_s = df["load_actual"].shift(24)
    df["price_roll24_mean"] = price_s.rolling(24, min_periods=24).mean()
    df["price_roll168_mean"] = price_s.rolling(168, min_periods=168).mean()
    df["price_roll24_std"] = price_s.rolling(24, min_periods=24).std()
    df["load_roll24_mean"] = load_s.rolling(24, min_periods=24).mean()
    df["load_roll168_mean"] = load_s.rolling(168, min_periods=168).mean()
    return df


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2015–17 train, 2018 test. Inclusive of 31 Dec 2017 23:00 Madrid."""
    train = frame.loc[: "2017-12-31"]
    test = frame.loc["2018-01-01":]
    return train, test
