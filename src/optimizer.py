"""Daily battery arbitrage LP.

Formulation (one day, T hours — 23/24/25 on DST)
-----------------------------------------------
Decision variables (continuous):
    c_t  charge from grid, MW, 0..P
    d_t  discharge to grid, MW, 0..P
    E_t  state of charge at the *start* of hour t, MWh, E_min..E_max
         plus E_T at end of the last hour

Physics (Δt = 1 h):
    E_{t+1} = E_t + η_c c_t - d_t / η_d
    η_c = η_d = sqrt(round_trip)     # losses split equally (assumption)

Daily cycle (so we don't drain the asset on day 1):
    E_T = E_0                        # E_0 free inside [E_min, E_max]

Throughput cap:
    sum_t d_t  <=  max_cycles * capacity     # MWh to grid

Objective (linprog minimises):
    sum_t price_t (c_t - d_t)  +  degradation * sum_t d_t

Simultaneous charge+discharge is feasible but never optimal when η < 1
and prices are positive — we do not need binaries.

Perfect foresight: `prices` = realised spot.
Realistic:         `prices` = our 24h forecast; P&L is then *settled*
                   against realised spot with the same (c, d).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.config import (
    ASSET_LIFE_YEARS,
    BATTERY_CAPEX_EUR_PER_KWH,
    BATTERY_CAPACITY_MWH,
    BATTERY_POWER_MW,
    DEGRADATION_COST_EUR_PER_MWH,
    MAX_CYCLES_PER_DAY,
    MIN_SOC_PCT,
    ROUND_TRIP_EFFICIENCY,
    WACC,
)


@dataclass(frozen=True)
class BatteryParams:
    power_mw: float = BATTERY_POWER_MW
    capacity_mwh: float = BATTERY_CAPACITY_MWH
    round_trip: float = ROUND_TRIP_EFFICIENCY
    min_soc_pct: float = MIN_SOC_PCT
    max_cycles_per_day: float = MAX_CYCLES_PER_DAY
    degradation_eur_per_mwh: float = DEGRADATION_COST_EUR_PER_MWH

    @property
    def eta(self) -> float:
        return float(self.round_trip ** 0.5)

    @property
    def e_min(self) -> float:
        return self.min_soc_pct * self.capacity_mwh

    @property
    def e_max(self) -> float:
        return self.capacity_mwh


DEFAULT_BATTERY = BatteryParams()


@dataclass
class DispatchResult:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray  # length T+1
    cash_eur: float
    degradation_eur: float
    profit_eur: float
    success: bool
    message: str = ""

def solve_day(prices, battery: BatteryParams = DEFAULT_BATTERY) -> DispatchResult:
    """Optimal charge/discharge for one day given a price vector."""
    prices = np.asarray(prices, dtype=float)
    if np.any(~np.isfinite(prices)):
        raise ValueError("prices contain NaN/inf")
    T = int(len(prices))
    if T < 2:
        raise ValueError("need at least 2 hours")

    P, eta, deg = battery.power_mw, battery.eta, battery.degradation_eur_per_mwh
    e_min, e_max = battery.e_min, battery.e_max
    n = 3 * T + 1  # c, d, E_0..E_T

    cobj = np.zeros(n)
    cobj[0:T] = prices                          # paying to charge
    cobj[T : 2 * T] = -prices + deg             # paid to discharge, minus deg

    A_eq = np.zeros((T + 1, n))
    b_eq = np.zeros(T + 1)
    for t in range(T):
        # E_{t+1} - E_t - eta c_t + d_t / eta = 0
        A_eq[t, 2 * T + t + 1] = 1.0
        A_eq[t, 2 * T + t] = -1.0
        A_eq[t, t] = -eta
        A_eq[t, T + t] = 1.0 / eta
    A_eq[T, 2 * T + T] = 1.0                    # E_T - E_0 = 0
    A_eq[T, 2 * T] = -1.0

    A_ub = np.zeros((1, n))
    A_ub[0, T : 2 * T] = 1.0
    b_ub = np.array([battery.max_cycles_per_day * battery.capacity_mwh])

    bounds = [(0.0, P)] * T + [(0.0, P)] * T + [(e_min, e_max)] * (T + 1)
    res = linprog(
        cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if not res.success:
        # Feasible fallback: do nothing, sit at mid SOC.
        mid = 0.5 * (e_min + e_max)
        return DispatchResult(
            charge_mw=np.zeros(T), discharge_mw=np.zeros(T),
            soc_mwh=np.full(T + 1, mid),
            cash_eur=0.0, degradation_eur=0.0, profit_eur=0.0,
            success=False, message=res.message,
        )

    x = res.x
    charge, discharge, soc = x[0:T], x[T : 2 * T], x[2 * T :]
    cash = float(np.dot(prices, discharge - charge))
    degradation = float(deg * discharge.sum())
    return DispatchResult(
        charge_mw=charge, discharge_mw=discharge, soc_mwh=soc,
        cash_eur=cash, degradation_eur=degradation,
        profit_eur=cash - degradation, success=True, message="ok",
    )


def settle(result: DispatchResult, actual_prices, battery: BatteryParams = DEFAULT_BATTERY) -> DispatchResult:
    """Apply a schedule to realised prices. Physics (SOC) unchanged."""
    actual = np.asarray(actual_prices, dtype=float)
    cash = float(np.dot(actual, result.discharge_mw - result.charge_mw))
    degradation = float(battery.degradation_eur_per_mwh * result.discharge_mw.sum())
    return DispatchResult(
        charge_mw=result.charge_mw, discharge_mw=result.discharge_mw,
        soc_mwh=result.soc_mwh, cash_eur=cash, degradation_eur=degradation,
        profit_eur=cash - degradation, success=result.success,
        message=result.message,
    )


def assert_invariants(result: DispatchResult, battery: BatteryParams = DEFAULT_BATTERY, atol: float = 1e-4) -> None:
    """SOC / power / energy-balance checks. Used by tests and the notebook."""
    c, d, e = result.charge_mw, result.discharge_mw, result.soc_mwh
    T = len(c)
    assert len(d) == T and len(e) == T + 1
    assert np.all(c >= -atol) and np.all(c <= battery.power_mw + atol)
    assert np.all(d >= -atol) and np.all(d <= battery.power_mw + atol)
    assert np.all(e >= battery.e_min - atol) and np.all(e <= battery.e_max + atol)
    eta = battery.eta
    for t in range(T):
        predicted = e[t] + eta * c[t] - d[t] / eta
        assert abs(predicted - e[t + 1]) < 1e-3, (t, predicted, e[t + 1])
    assert abs(e[0] - e[-1]) < 1e-3
    assert d.sum() <= battery.max_cycles_per_day * battery.capacity_mwh + atol


def backtest_daily(
    actual: pd.Series,
    schedule_on: pd.Series | None = None,
    battery: BatteryParams = DEFAULT_BATTERY,
) -> pd.DataFrame:
    """Day-by-day LP. If `schedule_on` is given, optimise on it and settle on `actual`."""
    actual = actual.astype(float).dropna()
    sched = actual if schedule_on is None else schedule_on.astype(float)
    rows = []
    hourly = []
    for day, act_day in actual.groupby(actual.index.date):
        opt_day = sched.reindex(act_day.index)
        if opt_day.isna().any() or act_day.isna().any():
            continue
        planned = solve_day(opt_day.values, battery)
        settled = settle(planned, act_day.values, battery)
        vwap_sell = (
            float(np.dot(act_day.values, settled.discharge_mw) / settled.discharge_mw.sum())
            if settled.discharge_mw.sum() > 1e-6 else np.nan
        )
        vwap_buy = (
            float(np.dot(act_day.values, settled.charge_mw) / settled.charge_mw.sum())
            if settled.charge_mw.sum() > 1e-6 else np.nan
        )
        rows.append(
            {
                "day": pd.Timestamp(day),
                "profit_eur": settled.profit_eur,
                "cash_eur": settled.cash_eur,
                "degradation_eur": settled.degradation_eur,
                "mwh_charge": float(settled.charge_mw.sum()),
                "mwh_discharge": float(settled.discharge_mw.sum()),
                "hours_charge": int((settled.charge_mw > 1.0).sum()),
                "hours_discharge": int((settled.discharge_mw > 1.0).sum()),
                "vwap_buy": vwap_buy,
                "vwap_sell": vwap_sell,
                "spread_eur": (vwap_sell - vwap_buy) if np.isfinite(vwap_sell) and np.isfinite(vwap_buy) else np.nan,
                "cycles": float(settled.discharge_mw.sum() / battery.capacity_mwh),
                "success": settled.success,
            }
        )
        idx = act_day.index
        hourly.append(
            pd.DataFrame(
                {
                    "charge_mw": settled.charge_mw,
                    "discharge_mw": settled.discharge_mw,
                    "soc_mwh": settled.soc_mwh[:-1],
                    "price_actual": act_day.values,
                    "price_used": opt_day.values,
                },
                index=idx,
            )
        )
    daily = pd.DataFrame(rows).set_index("day").sort_index()
    hourly_df = pd.concat(hourly).sort_index() if hourly else pd.DataFrame()
    return daily, hourly_df


def economics(
    annual_profit_eur: float,
    capacity_mwh: float = BATTERY_CAPACITY_MWH,
    capex_eur_per_kwh: float = BATTERY_CAPEX_EUR_PER_KWH,
    wacc: float = WACC,
    life_years: int = ASSET_LIFE_YEARS,
) -> dict:
    """Simple project finance on a constant annuity. Assumptions in config."""
    capex = capex_eur_per_kwh * capacity_mwh * 1000.0  # kWh
    if abs(wacc) < 1e-12:
        annuity = life_years
    else:
        annuity = (1.0 - (1.0 + wacc) ** (-life_years)) / wacc
    npv = -capex + annual_profit_eur * annuity
    payback = capex / annual_profit_eur if annual_profit_eur > 0 else np.inf
    return {
        "capex_eur": capex,
        "annual_profit_eur": annual_profit_eur,
        "npv_eur": npv,
        "payback_years": payback,
        "annuity_factor": annuity,
        "wacc": wacc,
        "life_years": life_years,
        "capex_eur_per_kwh": capex_eur_per_kwh,
    }


def with_battery(**overrides) -> BatteryParams:
    return replace(DEFAULT_BATTERY, **overrides)
