"""Page 5 — Dispatch. The LP on a single day."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.app_data import slice_day
from src.optimizer import DEFAULT_BATTERY, settle, solve_day
from src.ui import (
    AMBER,
    CHARGE,
    DISCHARGE,
    TEAL,
    apply_layout,
    callout,
    chart,
    forecasts_2018,
    fmt_eur,
    hero,
    kpis,
    market as load_market,
    need,
    page_boot,
)


def _schedule(actual: pd.Series, used: pd.Series):
    planned = solve_day(used.values)
    settled = settle(planned, actual.values)
    return planned, settled


def render() -> None:
    day = page_boot()
    mkt = need(load_market(), "market.parquet", "notebooks/01_eda.ipynb")
    piece = slice_day(mkt, day)
    if piece.empty:
        callout(f"No market rows for {day.isoformat()}.", "warn")
        st.stop()

    hero(
        "05 · Dispatch",
        "The LP is optimal and mute",
        "One Madrid-local day, three brains: omniscient spot, VOLTA’s P50, the TSO’s "
        "day-ahead. Physics are identical — 88% round-trip split √η, cyclic SOC, "
        "1 cycle/day, €3/MWh degradation. P&amp;L is always settled on realised spot.",
    )

    fc = forecasts_2018()
    fc_day = slice_day(fc, day) if fc is not None else piece.iloc[0:0]

    policy = st.radio(
        "Schedule on",
        ["VOLTA P50", "TSO day-ahead", "Perfect foresight"],
        horizontal=True,
        index=0 if not fc_day.empty else 2,
    )

    actual = piece["price_actual"]
    if policy == "Perfect foresight":
        used = actual
        policy_key = "perfect"
    elif policy == "TSO day-ahead":
        used = piece["price_da"]
        policy_key = "tso"
    else:
        if not fc_day.empty and "price_p50" in fc_day.columns:
            used = fc_day["price_p50"].reindex(actual.index)
            if used.isna().any():
                used = piece["price_da"]
                st.caption("P50 had holes — fell back to TSO DA for this day.")
        else:
            used = piece["price_da"]
            st.caption("No 2018 forecast file for this day — scheduling on TSO DA.")
        policy_key = "volta"

    planned, settled = _schedule(actual, used)

    kpis(
        [
            ("Settled P&L", fmt_eur(settled.profit_eur, 0), f"{policy} · cash {fmt_eur(settled.cash_eur, 0)}"),
            ("Discharged", f"{settled.discharge_mw.sum():.0f} MWh",
             f"{settled.discharge_mw.sum()/DEFAULT_BATTERY.capacity_mwh:.2f} cycles"),
            ("VWAP buy / sell",
             f"{_vwap(actual.values, settled.charge_mw):.1f} → {_vwap(actual.values, settled.discharge_mw):.1f}",
             "realised €/MWh"),
            ("Degradation", fmt_eur(settled.degradation_eur, 0), "€3 / MWh to grid"),
        ]
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Scatter(x=piece.index, y=actual, name="spot", line=dict(color=AMBER, width=2)),
        row=1, col=1, secondary_y=False,
    )
    if not np.allclose(used.values, actual.values):
        fig.add_trace(
            go.Scatter(x=piece.index, y=used, name="schedule on",
                       line=dict(color=TEAL, width=1.6, dash="dot")),
            row=1, col=1, secondary_y=False,
        )
    fig.add_trace(
        go.Bar(x=piece.index, y=settled.charge_mw, name="charge",
               marker_color=CHARGE, opacity=0.7),
        row=1, col=1, secondary_y=True,
    )
    fig.add_trace(
        go.Bar(x=piece.index, y=-settled.discharge_mw, name="discharge",
               marker_color=DISCHARGE, opacity=0.8),
        row=1, col=1, secondary_y=True,
    )
    soc = settled.soc_mwh[:-1]
    fig.add_trace(
        go.Scatter(x=piece.index, y=soc, name="SOC",
                   line=dict(color="#A78BFA", width=2), fill="tozeroy",
                   fillcolor="rgba(167,139,250,0.12)"),
        row=2, col=1,
    )
    fig.add_hline(y=DEFAULT_BATTERY.e_min, line_dash="dot", line_color="#64748B", row=2, col=1)
    fig.add_hline(y=DEFAULT_BATTERY.e_max, line_dash="dot", line_color="#64748B", row=2, col=1)
    fig.update_yaxes(title_text="€/MWh", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="MW", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="SOC MWh", row=2, col=1,
                     range=[0, DEFAULT_BATTERY.capacity_mwh * 1.05])
    apply_layout(fig, height=520, barmode="relative", title=f"{day.isoformat()} · {policy}")
    chart(fig)

    # three-way comparison for this day
    rows = []
    for name, series in [
        ("Perfect", actual),
        ("TSO", piece["price_da"]),
        ("VOLTA", used if policy_key == "volta" else (
            fc_day["price_p50"].reindex(actual.index)
            if not fc_day.empty and "price_p50" in fc_day.columns
            else piece["price_da"]
        )),
    ]:
        if series.isna().any():
            continue
        _, s = _schedule(actual, series)
        rows.append({
            "policy": name,
            "settled €": round(s.profit_eur, 0),
            "MWh out": round(s.discharge_mw.sum(), 1),
            "cycles": round(s.discharge_mw.sum() / DEFAULT_BATTERY.capacity_mwh, 2),
        })
    st.dataframe(pd.DataFrame(rows).set_index("policy"), width="stretch")

    callout(
        "<b>Assumptions an interviewer will ask.</b> η_c = η_d = √0.88 ≈ 0.938 (losses split equally). "
        "E_T = E_0 so we do not drain the asset on day 1. Throughput cap is 1.0 × capacity MWh to the grid. "
        "Simultaneous charge+discharge is feasible in the LP but never optimal when η&lt;1 and prices are positive."
    )


def _vwap(prices, mw) -> float:
    s = float(np.sum(mw))
    if s < 1e-6:
        return float("nan")
    return float(np.dot(prices, mw) / s)
