"""Decision-engine behaviour — the cards the app will render."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decisions import ACTIONS, breakeven_spread, decide_day
from src.optimizer import DEFAULT_BATTERY


def _hours(prices, tz="Europe/Madrid"):
    idx = pd.date_range("2018-06-01", periods=len(prices), freq="h", tz=tz)
    return pd.Series(prices, index=idx)


def test_flat_prices_are_all_hold():
    p = _hours(np.full(24, 50.0))
    d = decide_day(p)
    assert set(d.cards["action"]) == {"HOLD"}
    assert d.scale == 0.0 or d.cards["mw"].sum() == 0
    assert "BELOW_BREAKEVEN" in d.day_flags or d.cards["mw"].sum() == 0


def test_fat_spread_charges_then_discharges():
    prices = np.concatenate([np.full(12, 20.0), np.full(12, 80.0)])
    d = decide_day(_hours(prices))
    assert "CHARGE" in set(d.cards["action"])
    assert "DISCHARGE" in set(d.cards["action"])
    # night charge, evening discharge
    night = d.cards.iloc[:12]
    eve = d.cards.iloc[12:]
    assert night.loc[night["action"] == "CHARGE", "mw"].sum() > 0
    assert eve.loc[eve["action"] == "DISCHARGE", "mw"].sum() > 0
    assert d.expected_eur != 0.0
    assert d.scale == 1.0


def test_wide_quantile_band_shrinks_position():
    p50 = _hours(np.concatenate([np.full(12, 20.0), np.full(12, 80.0)]))
    # ~50% relative width → HARD
    p10 = p50 * 0.7
    p90 = p50 * 1.3
    fat = decide_day(p50)                       # no quantiles, full size
    thin = decide_day(p50, p10, p90)
    assert "HIGH_UNCERTAINTY" in thin.day_flags
    assert thin.scale < fat.scale
    assert thin.cards["mw"].sum() < fat.cards["mw"].sum()


def test_below_breakeven_holds():
    # €5 spread cannot cover 12% rt + €3 deg
    prices = np.concatenate([np.full(12, 50.0), np.full(12, 55.0)])
    d = decide_day(_hours(prices))
    assert d.scale == 0.0
    assert "BELOW_BREAKEVEN" in d.day_flags
    assert set(d.cards["action"]) == {"HOLD"}


def test_spike_flag_on_expensive_hour_not_a_rewrite():
    prices = np.concatenate([np.full(20, 40.0), np.full(4, 110.0)])
    d = decide_day(_hours(prices))
    spiked = d.cards[d.cards["flags"].str.contains("SPIKE")]
    assert len(spiked) >= 1
    # we do not zero a discharge just because it is a spike — we want that print
    if (spiked["action"] == "DISCHARGE").any():
        assert spiked.loc[spiked["action"] == "DISCHARGE", "mw"].sum() > 0


def test_card_schema_and_ranks():
    prices = np.concatenate([np.full(12, 20.0), np.full(12, 80.0)])
    d = decide_day(_hours(prices))
    for col in ("action", "mw", "confidence", "expected_eur", "why", "rank", "flags"):
        assert col in d.cards.columns
    assert set(d.cards["action"]).issubset(set(ACTIONS))
    assert d.cards["mw"].max() <= DEFAULT_BATTERY.power_mw + 1e-6
    # rank 1 is a money hour, not a HOLD
    top = d.cards.sort_values("rank").iloc[0]
    assert top["action"] in ("CHARGE", "DISCHARGE")
    assert top["why"]


def test_breakeven_formula_positive():
    be = breakeven_spread(50.0)
    # 50/0.88 - 50 + 3 ≈ 9.8
    assert 8.0 < be < 12.0
