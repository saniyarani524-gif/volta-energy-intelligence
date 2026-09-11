"""Page 6 — Ledger. Settled P&L and the honest economics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import economics
from src.ui import (
    AMBER,
    ROSE,
    TEAL,
    apply_layout,
    callout,
    chart,
    dispatch_daily,
    fmt_eur,
    fmt_eur_k,
    hero,
    kpis,
    need,
    page_boot,
)


def render() -> None:
    page_boot()
    daily = need(dispatch_daily(), "dispatch_*_daily.parquet", "notebooks/03_dispatch.ipynb")

    hero(
        "06 · Ledger",
        "The battery does not pay back on 2015–18 Spanish spreads",
        "That sentence is the product. Perfect foresight is a ceiling, not a strategy. "
        "VOLTA keeps about four fifths of that ceiling on 2016–18. Capex is 2018-era "
        "turnkey Li-ion at €350/kWh — not a 2024 pack price.",
    )

    perfect = daily[daily["policy"] == "perfect"]
    volta = daily[daily["policy"] == "volta"]
    tso = daily[daily["policy"] == "tso"]

    p_sum = float(perfect["profit_eur"].sum())
    p_ann = p_sum / max(perfect.index.year.nunique(), 1)
    v_sum = float(volta["profit_eur"].sum()) if len(volta) else 0.0
    t_sum = float(tso["profit_eur"].sum()) if len(tso) else 0.0
    p_overlap = perfect.loc[perfect.index.year >= 2016, "profit_eur"].sum() if len(perfect) else np.nan
    keep = v_sum / p_overlap if p_overlap else np.nan

    e_p = economics(p_ann)
    v_years = max(volta.index.year.nunique(), 1) if len(volta) else 3
    e_v = economics(v_sum / v_years) if len(volta) else None

    kpis(
        [
            ("Perfect 2015–18", fmt_eur_k(p_sum), f"~{fmt_eur_k(p_ann)} / year ceiling"),
            ("VOLTA 2016–18", fmt_eur_k(v_sum), f"{keep:.0%} of same-window ceiling" if keep == keep else ""),
            ("TSO 2016–18", fmt_eur_k(t_sum), "schedule on the operator’s price"),
            ("Payback @ perfect", f"{e_p['payback_years']:.0f} years",
             f"NPV {fmt_eur_k(e_p['npv_eur'])} · capex {fmt_eur_k(e_p['capex_eur'])}"),
        ]
    )

    callout(
        f"€70m capex, 15-year life, 7% WACC. Even the omniscient battery returns "
        f"{fmt_eur_k(p_ann)}/year and an NPV of <b>{fmt_eur_k(e_p['npv_eur'])}</b>. "
        "Spreads on the peninsula in this window do not buy a 200 MWh pack. "
        "The desk is still the right object — the honest conclusion is the interview.",
        "danger",
    )

    fig = go.Figure()
    for name, df, color in [
        ("perfect", perfect, AMBER),
        ("volta", volta, TEAL),
        ("tso", tso, ROSE),
    ]:
        if df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["profit_eur"].cumsum(), name=name,
                line=dict(color=color, width=2),
            )
        )
    apply_layout(fig, height=400, title="Cumulative settled P&L")
    fig.update_yaxes(title="€")
    chart(fig)

    left, right = st.columns(2)
    with left:
        rows = []
        for name, df in [("perfect", perfect), ("volta", volta), ("tso", tso)]:
            if df.empty:
                continue
            g = df["profit_eur"].groupby(df.index.year).sum()
            for y, val in g.items():
                rows.append({"year": int(y), "policy": name, "€": val})
        if rows:
            wide = pd.DataFrame(rows).pivot(index="year", columns="policy", values="€")
            fig = go.Figure()
            colors = {"perfect": AMBER, "volta": TEAL, "tso": ROSE}
            for col in wide.columns:
                fig.add_trace(go.Bar(x=wide.index.astype(str), y=wide[col], name=col,
                                     marker_color=colors.get(col, TEAL)))
            apply_layout(fig, height=340, barmode="group", title="Annual settled P&L")
            chart(fig)

    with right:
        st.markdown("<div class='eyebrow'>Project finance (constant annuity)</div>", unsafe_allow_html=True)
        capex = st.slider("Capex €/kWh (2018-era turnkey)", 100, 500, 350, step=10)
        wacc = st.slider("WACC", 0.03, 0.12, 0.07, step=0.01, format="%.2f")
        life = st.slider("Life (years)", 8, 25, 15)
        src = st.radio("Annuity from", ["Perfect annual", "VOLTA annual"], horizontal=True)
        ann = p_ann if src.startswith("Perfect") else (v_sum / v_years)
        e = economics(ann, capex_eur_per_kwh=float(capex), wacc=float(wacc), life_years=int(life))
        st.markdown(
            f"""
            <div class='kpi-grid'>
              <div class='kpi'><div class='label'>Capex</div>
                <div class='value'>{fmt_eur_k(e['capex_eur'])}</div>
                <div class='sub'>200 MWh × €{capex}/kWh</div></div>
              <div class='kpi'><div class='label'>NPV</div>
                <div class='value'>{fmt_eur_k(e['npv_eur'])}</div>
                <div class='sub'>{life}y @ {wacc:.0%} WACC</div></div>
              <div class='kpi'><div class='label'>Payback</div>
                <div class='value'>{e['payback_years']:.1f}y</div>
                <div class='sub'>simple · no residual</div></div>
              <div class='kpi'><div class='label'>Annuity used</div>
                <div class='value'>{fmt_eur_k(ann)}/y</div>
                <div class='sub'>repeats the backtest forever</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Slider does not re-run the LP. It re-prices the same annual cashflow. That is the point.")

    if "spread_eur" in perfect.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=perfect["spread_eur"].dropna(), name="perfect spread",
                                   marker_color=TEAL, nbinsx=40, opacity=0.85))
        apply_layout(fig, height=280, title="Daily realised spread €/MWh (perfect policy)")
        fig.update_xaxes(title="€/MWh")
        chart(fig)
