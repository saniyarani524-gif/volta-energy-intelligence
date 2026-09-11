"""Page 4 — Forecast desk. Beat the TSO on price, not load."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import mae_table_2018, slice_day
from src.ui import (
    AMBER,
    ROSE,
    TEAL,
    apply_layout,
    callout,
    chart,
    forecasts_2018,
    hero,
    importance,
    kpis,
    need,
    page_boot,
    walkforward,
)


def render() -> None:
    day = page_boot()
    fc = need(forecasts_2018(), "test_2018_forecasts.parquet", "notebooks/02_forecasting.ipynb")

    tbl = mae_table_2018(fc)
    tso = float(tbl.loc["TSO day-ahead", "MAE"]) if "TSO day-ahead" in tbl.index else np.nan
    ours = float(tbl.loc["VOLTA XGB", "MAE"]) if "VOLTA XGB" in tbl.index else np.nan
    naive = float(tbl.loc["Naive-24h", "MAE"]) if "Naive-24h" in tbl.index else np.nan
    lift = (tso - ours) / tso * 100 if tso and ours == ours else np.nan

    hero(
        "04 · Forecast",
        "Beat the TSO on price. Do not touch load.",
        "The operator already has demand to ~1% MAPE. Their day-ahead <i>price</i> number "
        "is a different animal — biased cheap, worst at the evening peak. That is the claim.",
    )

    kpis(
        [
            ("2018 TSO MAE", f"€{tso:.2f}", "the number we have to beat"),
            ("2018 VOLTA MAE", f"€{ours:.2f}", f"{lift:.0f}% better than the TSO" if lift == lift else ""),
            ("Naive-24h", f"€{naive:.2f}", "yesterday same hour already beats the TSO"),
            ("P10–P90 cover", f"{((fc.price_actual.between(fc.price_p10, fc.price_p90)).mean()*100):.0f}%",
             "target 80% — slightly conservative is what a battery wants"),
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=list(tbl.index),
            y=tbl["MAE"],
            marker_color=[ROSE if "TSO" in i else (AMBER if "Naive" in i else TEAL) for i in tbl.index],
            text=[f"€{v:.2f}" for v in tbl["MAE"]],
            textposition="outside",
        )
    )
    apply_layout(fig, height=320, title="2018 price MAE €/MWh — temporal split, no shuffle")
    fig.update_yaxes(title="MAE", range=[0, tbl["MAE"].max() * 1.3])
    chart(fig)
    st.dataframe(tbl.round(2), width="stretch")

    tab_zoom, tab_q, tab_err, tab_imp = st.tabs(
        ["Two-week zoom", "Quantile band", "Error by hour", "Importance"]
    )

    with tab_zoom:
        zoom = fc.loc["2018-01-15":"2018-01-28"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=zoom.index, y=zoom["price_actual"], name="spot",
                                 line=dict(color=AMBER, width=1.8)))
        fig.add_trace(go.Scatter(x=zoom.index, y=zoom["price_pred"], name="VOLTA",
                                 line=dict(color=TEAL, width=1.6)))
        fig.add_trace(go.Scatter(x=zoom.index, y=zoom["price_da"], name="TSO DA",
                                 line=dict(color="#64748B", width=1.2, dash="dot")))
        apply_layout(fig, height=380, title="15–28 Jan 2018 — where the TSO actually fails")
        fig.update_yaxes(title="€/MWh")
        chart(fig)

    with tab_q:
        piece = slice_day(fc, day)
        if piece.empty:
            piece = fc.loc["2018-01-25":"2018-01-25"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(piece.index) + list(piece.index[::-1]),
            y=list(piece["price_p90"]) + list(piece["price_p10"][::-1]),
            fill="toself", fillcolor="rgba(45,212,191,0.18)",
            line=dict(width=0), name="P10–P90", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=piece.index, y=piece["price_p50"], name="P50",
                                 line=dict(color=TEAL, width=2)))
        fig.add_trace(go.Scatter(x=piece.index, y=piece["price_actual"], name="spot",
                                 line=dict(color=AMBER, width=2)))
        apply_layout(fig, height=360, title=f"{day.isoformat()} · quantile band (P10 / P50 / P90)")
        fig.update_yaxes(title="€/MWh")
        chart(fig)
        callout(
            "An 80% interval that actually covers ~81% of 2018 is slightly conservative. "
            "Phase 4 reads the relative width (P90−P10)/P50 and cuts the <b>whole day</b> "
            "to 50% or 25% — never hour-by-hour, which would break SOC."
        )

    with tab_err:
        err = (fc["price_pred"] - fc["price_actual"]).abs()
        tso_err = (fc["price_da"] - fc["price_actual"]).abs()
        by_h = pd.DataFrame({"VOLTA": err.groupby(err.index.hour).mean(),
                             "TSO": tso_err.groupby(tso_err.index.hour).mean()})
        fig = go.Figure()
        fig.add_trace(go.Bar(x=by_h.index, y=by_h["TSO"], name="TSO", marker_color=ROSE, opacity=0.75))
        fig.add_trace(go.Bar(x=by_h.index, y=by_h["VOLTA"], name="VOLTA", marker_color=TEAL))
        apply_layout(fig, height=360, barmode="group", title="2018 MAE by hour — the peak is where it pays")
        fig.update_xaxes(title="hour", dtick=1)
        fig.update_yaxes(title="€/MWh")
        chart(fig)

        wf = walkforward()
        if wf is not None:
            wf_err = (wf["pred"] - wf["actual"]).abs()
            monthly = wf_err.groupby([wf_err.index.year, wf_err.index.month]).mean()
            labels = [f"{y}-{m:02d}" for y, m in monthly.index]
            fig = go.Figure(go.Bar(x=labels, y=monthly.values, marker_color=TEAL))
            apply_layout(fig, height=300, title="Walk-forward MAE by month (2016–18 expanding window)")
            fig.update_xaxes(dtick=4, tickangle=-45)
            chart(fig)

    with tab_imp:
        imp = importance()
        if imp is None:
            callout("XGBoost JSON not on disk (models/ is git-ignored). Run notebook 02 to write it.", "warn")
        else:
            top = imp.tail(15)
            fig = go.Figure(
                go.Bar(x=top.values, y=top.index, orientation="h", marker_color=TEAL)
            )
            apply_layout(fig, height=420, title="Gain importance — lags legal at ≥24h, TSO DA is fair game")
            chart(fig)
            callout(
                "Dominance of <span class='mono'>price_lag24</span> / <span class='mono'>price_lag168</span> "
                "is the market being sticky — and we are allowed to use those lags. "
                "Realised wind/load at hour t are not in this list. That is the leakage contract."
            )

    callout(
        "<b>Load, honestly.</b> TSO load MAE is ~270 MW (~0.9% MAPE). Our XGB is slightly worse. "
        "The desk uses the operator’s <span class='mono'>load_da</span> and spends the modelling budget on price."
    )
