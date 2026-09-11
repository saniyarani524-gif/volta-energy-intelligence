"""Invariants for the daily battery LP. These are the interview."""
from __future__ import annotations

import numpy as np
import pytest

from src.optimizer import (
    DEFAULT_BATTERY,
    assert_invariants,
    solve_day,
    settle,
    with_battery,
)


def test_toy_charges_cheap_discharges_expensive():
    prices = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])
    r = solve_day(prices)
    assert r.success
    assert_invariants(r)
    # first half mostly charge, second half mostly discharge
    assert r.charge_mw[:3].sum() > r.charge_mw[3:].sum()
    assert r.discharge_mw[3:].sum() > r.discharge_mw[:3].sum()
    assert r.profit_eur > 0


def test_soc_never_outside_bounds():
    rng = np.random.default_rng(7)
    for _ in range(15):
        prices = 40 + 30 * rng.random(24)
        r = solve_day(prices)
        assert_invariants(r)


def test_flat_prices_do_nothing_after_degradation():
    r = solve_day(np.full(24, 50.0))
    assert r.success
    assert_invariants(r)
    # round-trip loss + €3/MWh makes cycling unprofitable when the spread is 0
    assert r.discharge_mw.sum() < 1.0
    assert abs(r.profit_eur) < 1.0


def test_power_and_energy_caps():
    r = solve_day(np.concatenate([np.full(12, 1.0), np.full(12, 200.0)]))
    assert_invariants(r)
    assert r.charge_mw.max() <= DEFAULT_BATTERY.power_mw + 1e-6
    assert r.discharge_mw.max() <= DEFAULT_BATTERY.power_mw + 1e-6
    assert r.discharge_mw.sum() <= DEFAULT_BATTERY.capacity_mwh + 1e-4


def test_dst_23_and_25_hour_days():
    for T in (23, 25):
        prices = np.linspace(20, 80, T)
        r = solve_day(prices)
        assert r.success
        assert_invariants(r)
        assert len(r.charge_mw) == T
        assert len(r.soc_mwh) == T + 1


def test_settle_uses_actual_prices_not_forecast():
    forecast = np.concatenate([np.full(12, 10.0), np.full(12, 90.0)])
    actual = np.concatenate([np.full(12, 90.0), np.full(12, 10.0)])  # inverted
    planned = solve_day(forecast)
    settled = settle(planned, actual)
    # scheduled against a lie → should make less than (or lose vs) doing it on actuals
    oracle = solve_day(actual)
    assert settled.profit_eur < oracle.profit_eur


def test_smaller_battery_still_feasible():
    bat = with_battery(power_mw=10.0, capacity_mwh=20.0)
    r = solve_day(np.linspace(20, 80, 24), battery=bat)
    assert r.success
    assert_invariants(r, battery=bat)
