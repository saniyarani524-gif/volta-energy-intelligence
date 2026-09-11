"""Page 2 — Market pulse. Four years of Spanish spot."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import slice_day
from src.ui import (
    AMBER,
    TEAL,
    apply_layout,
    callout,
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
    mkt = need(load_market(), "market.parquet", "notebooks/01_eda.ipynb")

    hero(
        "02 · Market",
        "Four years of Spanish spot",
        "Demand barely moves. Price does. 2016 is the cheap hydro year; 2017 prints the "
        "spikes. The TSO’s own day-ahead price forecast is the benchmark we have to beat — "
        "not because it is good, because it is the operator’s number.",
    )

    price = mkt["price_actual"]
    tso_mae = float((mkt["price_da"] - price).abs().mean())
    kpis(
        [
            ("Hours", f"{len(mkt):,}", f"{mkt.index.min().date()} → {mkt.index.max().date()}"),
            ("Mean price", fmt_eur(float(price.mean()), 1) + "/MWh", f"max {fmt_eur(float(price.max()), 1)}"),
            ("Mean load", f"{mkt['load_actual'].mean()/1000:.1f} GW", "almost flat year to year"),
            ("TSO price MAE", f"€{tso_mae:.2f}", "full sample · worst at the evening peak"),
        ]
    )

    yearly = price.groupby(price.index.year).mean()
    years = [str(y) for y in yearly.index]
    fig = go.Figure(
        go.Bar(
            x=years,
            y=yearly.values,
            marker_color=[TEAL, "#1F6F66", AMBER, "#B45309"],
            text=[f"€{v:.0f}" for v in yearly.values],
            textposition="outside",
        )
    )
    apply_layout(fig, height=280, title="Annual mean spot €/MWh — 2016 is the cheap year")
    fig.update_yaxes(title="€/MWh", range=[0, max(yearly.values) * 1.25])
    chart(fig)

    tab_ts, tab_heat, tab_day = st.tabs(["Daily tape", "Hour × month", "This day"])

    with tab_ts:
        daily = mkt[["price_actual", "price_da", "load_actual"]].resample("D").mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily.index, y=daily["price_actual"], name="spot",
                line=dict(color=AMBER, width=1.1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily.index, y=daily["price_da"], name="TSO DA",
                line=dict(color=TEAL, width=1.0), opacity=0.75,
            )
        )
        apply_layout(fig, height=380, title="Daily mean price — actual vs TSO day-ahead")
        fig.update_yaxes(title="€/MWh")
        chart(fig)

        fig = go.Figure(
            go.Scatter(
                x=daily.index,
                y=daily["load_actual"] / 1000,
                name="load",
                line=dict(color="#38BDF8", width=1.1),
            )
        )
        apply_layout(fig, height=280, title="Daily mean load (GW) — the boring series, on purpose")
        fig.update_yaxes(title="GW")
        chart(fig)

    with tab_heat:
        tmp = mkt.copy()
        tmp["hour"] = tmp.index.hour
        tmp["month"] = tmp.index.month
        pivot = tmp.pivot_table(index="month", columns="hour", values="price_actual", aggfunc="mean")
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                colorscale=[
                    [0, "#0B0E13"],
                    [0.35, "#134E4A"],
                    [0.6, TEAL],
                    [0.8, AMBER],
                    [1, "#FB7185"],
                ],
                colorbar=dict(title="€/MWh"),
                hovertemplate="hour %{x} · %{y}<br>€%{z:.1f}<extra></extra>",
            )
        )
        apply_layout(fig, height=420, title="Mean spot by month × hour — two orthogonal spreads")
        fig.update_xaxes(title="hour", dtick=2)
        chart(fig)
        callout(
            "Night vs evening is the battery’s bread. Winter vs summer is the other axis. "
            "A 2-hour asset lives on the intra-day spread; the seasonal one is a bonus."
        )

    with tab_day:
        piece = slice_day(mkt, day)
        if piece.empty:
            callout("No rows this day.", "warn")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=piece.index, y=piece["price_actual"], name="spot",
                                     line=dict(color=AMBER, width=2.4)))
            fig.add_trace(go.Scatter(x=piece.index, y=piece["price_da"], name="TSO DA",
                                     line=dict(color=TEAL, width=1.6, dash="dot")))
            apply_layout(fig, height=360, title=f"{day.isoformat()} · hourly price")
            fig.update_yaxes(title="€/MWh")
            chart(fig)
            lo, hi = float(piece["price_actual"].min()), float(piece["price_actual"].max())
            st.caption(
                f"Range €{lo:.1f}–€{hi:.1f} · spread €{hi-lo:.1f}/MWh · "
                f"load {piece['load_actual'].mean()/1000:.1f} GW mean"
            )

    by_year = mkt.groupby(mkt.index.year)["price_actual"].agg(["mean", "std", "min", "max"])
    by_year.columns = ["mean €", "std", "min", "max"]
    st.dataframe(by_year.round(2), width="stretch")
