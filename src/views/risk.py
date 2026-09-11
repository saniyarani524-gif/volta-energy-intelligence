"""Page 8 — Risk. Flags, scale, and a live what-if battery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import slice_day
from src.config import SPIKE_EUR, UNCERTAINTY_HARD, UNCERTAINTY_SOFT
from src.decisions import breakeven_spread
from src.optimizer import DEFAULT_BATTERY, settle, solve_day, with_battery
from src.ui import (
    AMBER,
    CHARGE,
    DISCHARGE,
    HOLD,
    ROSE,
    TEAL,
    apply_layout,
    callout,
    cards as load_cards,
    chart,
    fmt_eur,
    hero,
    kpis,
    market as load_market,
    need,
    page_boot,
)


def render() -> None:
    day = page_boot()
    hero(
        "08 · Risk",
        "Flags, not silent rewrites",
        f"Relative quantile width ≥ {UNCERTAINTY_SOFT:.0%} cuts the day in half; "
        f"≥ {UNCERTAINTY_HARD:.0%} cuts to 25%. Spread below round-trip + €3 sits the day out. "
        f"A print ≥ €{SPIKE_EUR:.0f}/MWh is a SPIKE flag — we still want that discharge.",
    )

    cards = load_cards()
    if cards is not None and not cards.empty:
        day_flags = cards.groupby(cards["day"])["flags"].agg(
            lambda s: "|".join(sorted({f for v in s.astype(str) for f in v.split("|") if f}))
        )
        scale = cards.groupby(cards["day"])["scale"].first()
        actions = cards["action"].value_counts()
        kpis(
            [
                ("2018 HOLD hours", f"{int(actions.get('HOLD', 0)):,}",
                 f"{int(actions.get('CHARGE', 0))} charge · {int(actions.get('DISCHARGE', 0))} discharge"),
                ("Days sat out", f"{int((scale == 0).sum())}", "BELOW_BREAKEVEN · scale 0"),
                ("High-uncertainty days",
                 f"{int(day_flags.str.contains('HIGH_UNCERTAINTY').sum())}",
                 "scale 0.50 or 0.25"),
                ("Mean day scale", f"{scale.mean():.0%}", "most hours you do nothing"),
            ]
        )

        left, right = st.columns(2)
        with left:
            fig = go.Figure(
                go.Bar(
                    x=list(actions.index),
                    y=list(actions.values),
                    marker_color=[
                        {"CHARGE": CHARGE, "DISCHARGE": DISCHARGE, "HOLD": HOLD}.get(a, TEAL)
                        for a in actions.index
                    ],
                    text=[str(int(v)) for v in actions.values],
                    textposition="outside",
                )
            )
            apply_layout(fig, height=300, title="2018 hours by action")
            chart(fig)
        with right:
            fig = go.Figure(go.Histogram(x=scale.values, marker_color=TEAL, nbinsx=8))
            apply_layout(fig, height=300, title="Day-level scale (2018)")
            fig.update_xaxes(title="scale")
            chart(fig)

        cal = scale.copy()
        cal.index = pd.to_datetime(cal.index)
        pivot = pd.DataFrame({"scale": cal.values, "dow": cal.index.dayofweek, "week": cal.index.isocalendar().week.values,
                              "month": cal.index.month})
        # month × day-of-month heatmap of scale
        tmp = pd.DataFrame({"scale": scale.values}, index=pd.to_datetime(scale.index))
        tmp["month"] = tmp.index.month
        tmp["dom"] = tmp.index.day
        heat = tmp.pivot_table(index="month", columns="dom", values="scale", aggfunc="first")
        fig = go.Figure(
            go.Heatmap(
                z=heat.values,
                x=list(heat.columns),
                y=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                colorscale=[
                    [0, "#1C1917"],
                    [0.25, ROSE],
                    [0.5, AMBER],
                    [1.0, TEAL],
                ],
                zmin=0, zmax=1,
                colorbar=dict(title="scale"),
                hovertemplate="%{y} %{x}<br>scale %{z:.2f}<extra></extra>",
            )
        )
        apply_layout(fig, height=380, title="2018 calendar · day scale (black = sat out)")
        chart(fig)
    else:
        callout("Run notebook 04 to cache 2018 cards. The what-if below still works.", "warn")

    st.markdown("<div class='eyebrow' style='margin-top:8px'>What-if battery · this day, live LP</div>",
                unsafe_allow_html=True)
    mkt = need(load_market(), "market.parquet", "notebooks/01_eda.ipynb")
    piece = slice_day(mkt, day)
    if piece.empty:
        callout("No prices this day.", "warn")
        return

    c1, c2, c3, c4 = st.columns(4)
    power = c1.slider("Power MW", 10, 400, int(DEFAULT_BATTERY.power_mw), step=10)
    hours = c2.slider("Duration h", 1, 8, 2)
    rte = c3.slider("Round-trip %", 70, 96, int(DEFAULT_BATTERY.round_trip * 100))
    deg = c4.slider("Degradation €/MWh", 0, 15, int(DEFAULT_BATTERY.degradation_eur_per_mwh))
    cap = power * hours
    bat = with_battery(
        power_mw=float(power),
        capacity_mwh=float(cap),
        round_trip=rte / 100.0,
        degradation_eur_per_mwh=float(deg),
    )
    planned = solve_day(piece["price_actual"].values, battery=bat)
    settled = settle(planned, piece["price_actual"].values, battery=bat)
    be = breakeven_spread(float(piece["price_actual"].min()), bat)
    kpis(
        [
            ("This-day P&L (perfect)", fmt_eur(settled.profit_eur, 0),
             f"{cap:.0f} MWh pack · {rte}% RTE"),
            ("MWh to grid", f"{settled.discharge_mw.sum():.0f}",
             f"{settled.discharge_mw.sum()/cap:.2f} cycles"),
            ("Breakeven at floor", f"€{be:.1f}/MWh",
             "P_buy × (1/η − 1) + deg"),
            ("SOC close", f"{planned.soc_mwh[0]:.0f} → {planned.soc_mwh[-1]:.0f} MWh",
             "cyclic constraint"),
        ]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=piece.index, y=piece["price_actual"], name="spot",
                             line=dict(color=AMBER, width=2), yaxis="y1"))
    fig.add_trace(go.Bar(x=piece.index, y=settled.charge_mw, name="charge",
                         marker_color=CHARGE, opacity=0.7, yaxis="y2"))
    fig.add_trace(go.Bar(x=piece.index, y=-settled.discharge_mw, name="discharge",
                         marker_color=DISCHARGE, opacity=0.8, yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="€/MWh"),
        yaxis2=dict(title="MW", overlaying="y", side="right", showgrid=False),
        barmode="relative",
    )
    apply_layout(fig, height=340, title=f"{day.isoformat()} · perfect foresight on the slider battery")
    chart(fig)

    buys = np.linspace(10, 90, 17)
    fig = go.Figure(
        go.Scatter(
            x=buys,
            y=[breakeven_spread(float(b), bat) for b in buys],
            mode="lines+markers",
            line=dict(color=TEAL, width=2),
            name="breakeven spread",
        )
    )
    apply_layout(fig, height=280, title="Breakeven (sell − buy) vs buy price")
    fig.update_xaxes(title="buy €/MWh")
    fig.update_yaxes(title="need €/MWh")
    chart(fig)
    callout(
        "P&amp;L on this page is perfect-foresight for the slider — it is a ceiling for "
        "<i>this</i> day and <i>this</i> pack, not a new 3-year backtest. Re-running 1,000+ daily LPs "
        "on every slider tick would lie about interactivity."
    )
