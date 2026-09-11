"""Leakage guards — these tests are the interview.

If a future-you adds price_lag1 or gen_wind (same-hour) to PRICE_FEATURES,
this fails on purpose.
"""
from __future__ import annotations

import re

from src.features import (
    LOAD_FEATURES,
    PRICE_FEATURES,
    PRICE_FEATURES_NO_DA,
    REALISED_NOW,
)


def test_price_features_exclude_realised_nowcast():
    assert REALISED_NOW.isdisjoint(PRICE_FEATURES)


def test_load_features_exclude_realised_nowcast():
    # load_actual is the target, not a feature
    assert REALISED_NOW.isdisjoint(LOAD_FEATURES)


def test_minimum_realised_lag_is_24h():
    for name in (*PRICE_FEATURES, *LOAD_FEATURES):
        m = re.search(r"lag(\d+)$", name)
        if m:
            assert int(m.group(1)) >= 24, name


def test_ablation_drops_only_price_da():
    assert "price_da" not in PRICE_FEATURES_NO_DA
    assert set(PRICE_FEATURES) - set(PRICE_FEATURES_NO_DA) == {"price_da"}


def test_tso_columns_are_present():
    for col in ("price_da", "load_da", "forecast_solar_da", "forecast_wind_da", "net_load_da"):
        assert col in PRICE_FEATURES
