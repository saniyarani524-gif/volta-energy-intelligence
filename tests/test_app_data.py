"""Smoke for the desk loaders. Skip when the notebook artifacts are absent."""
from __future__ import annotations

from datetime import date

import pytest

from src.app_data import (
    DEFAULT_DAY,
    clamp_day,
    mae_table_2018,
    read_cards,
    read_forecasts_2018,
    read_market,
    slice_day,
)


def test_clamp_day():
    span = (date(2015, 1, 1), date(2018, 12, 31))
    assert clamp_day(date(2010, 1, 1), span) == date(2015, 1, 1)
    assert clamp_day(date(2020, 1, 1), span) == date(2018, 12, 31)
    assert clamp_day(DEFAULT_DAY, span) == DEFAULT_DAY


def test_market_or_skip():
    m = read_market()
    if m is None:
        pytest.skip("market.parquet not built")
    assert len(m) == 35064
    assert "price_actual" in m.columns
    day = slice_day(m, DEFAULT_DAY)
    assert 23 <= len(day) <= 25


def test_2018_mae_or_skip():
    fc = read_forecasts_2018()
    if fc is None:
        pytest.skip("forecast cache missing")
    tbl = mae_table_2018(fc)
    assert tbl.loc["VOLTA XGB", "MAE"] < tbl.loc["TSO day-ahead", "MAE"]
    assert tbl.loc["VOLTA XGB", "MAE"] < 5.0


def test_cards_schema_or_skip():
    c = read_cards()
    if c is None:
        pytest.skip("cards cache missing")
    for col in ("action", "mw", "why", "flags", "expected_eur"):
        assert col in c.columns
    assert set(c["action"]).issubset({"CHARGE", "DISCHARGE", "HOLD"})


def test_finder_sees_sandbox_parquet():
    from src.market_io import find_file, market_parquet

    p = market_parquet()
    assert p is not None, "sandbox should have data/processed/market.parquet"
    assert p.name == "market.parquet"
    fc = find_file("cache", "test_2018_forecasts.parquet")
    assert fc is not None
