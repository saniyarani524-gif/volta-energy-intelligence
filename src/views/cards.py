"""Page 7 — Cards. CHARGE / DISCHARGE / HOLD with a why."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.app_data import slice_day
from src.decisions import decide_day
from src.ui import (
    AMBER,
    CHARGE,
    DISCHARGE,
    TEAL,
    action_badge,
    apply_layout,
    callout,
    chart,
    cards as load_cards,
    forecasts_2018,
    fmt_eur,
    hero,
    kpis,
    market as load_market,
    page_boot,
)


def _cards_for(day):
    cached = load_cards()
    if cached is not None:
        piece = slice_day(cached, day)
        if not piece.empty:
            return piece, "cache 2018"
    fc = forecasts_2018()
    if fc is not None:
        piece = slice_day(fc, day)
        if not piece.empty and "price_p50" in piece:
            d = decide_day(piece["price_p50"], piece.get("price_p10"), piece.get("price_p90"))
            out = d.cards.copy()
            out["scale"] = d.scale
            out["day_expected_eur"] = d.expected_eur
            out["day_spread_eur"] = d.spread_eur
            return out, "live P50"
    mkt = load_market()
    if mkt is None:
        return None, ""
    piece = slice_day(mkt, day)
    if piece.empty:
        return None, ""
    d = decide_day(piece["price_actual"])
    out = d.cards.copy()
    out["scale"] = d.scale
    out["day_expected_eur"] = d.expected_eur
    out["day_spread_eur"] = d.spread_eur
    return out, "live on realised spot (no forecast)"


def render() -> None:
    day = page_boot()
    hero(
        "07 · Cards",
        "The why is the product",
        "A recruiter should be able to disagree with the trade and still see why the desk "
        "wanted it. Confidence is the P10–P90 width. Flags never silently rewrite physics "
        "except through the day-level scale.",
    )

    cards, source = _cards_for(day)
    if cards is None or cards.empty:
        callout(
            f"No cards for {day.isoformat()}. Run notebook 04 for 2018, or pick another day "
            "and the engine will solve live.",
            "warn",
        )
        st.stop()

    scale = float(cards["scale"].iloc[0]) if "scale" in cards else 1.0
    expected = float(cards["expected_eur"].sum())
    spread = float(cards["day_spread_eur"].iloc[0]) if "day_spread_eur" in cards else float("nan")
    counts = cards["action"].value_counts()
    kpis(
        [
            ("Source", source, day.isoformat()),
            ("Scale", f"{scale:.0%}", "whole day · SOC stays feasible"),
            ("Expected €", fmt_eur(expected, 0), f"spread {spread:.1f} €/MWh" if spread == spread else ""),
            ("Mix",
             f"{int(counts.get('CHARGE', 0))}C / {int(counts.get('DISCHARGE', 0))}D / {int(counts.get('HOLD', 0))}H",
             "hours"),
        ]
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "price_p10" in cards and "price_p90" in cards:
        fig.add_trace(
            go.Scatter(
                x=list(cards.index) + list(cards.index[::-1]),
                y=list(cards["price_p90"]) + list(cards["price_p10"][::-1]),
                fill="toself", fillcolor="rgba(45,212,191,0.14)",
                line=dict(width=0), name="P10–P90", hoverinfo="skip",
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(x=cards.index, y=cards["price_p50"], name="P50",
                   line=dict(color=TEAL, width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=cards.index, y=cards["mw"].where(cards["action"] == "CHARGE", 0),
               name="charge", marker_color=CHARGE, opacity=0.7),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(x=cards.index, y=-cards["mw"].where(cards["action"] == "DISCHARGE", 0),
               name="discharge", marker_color=DISCHARGE, opacity=0.8),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="€/MWh", secondary_y=False)
    fig.update_yaxes(title_text="MW", secondary_y=True, showgrid=False)
    apply_layout(fig, height=360, barmode="relative", title="P50 + band vs scaled action")
    chart(fig)

    filt = st.multiselect("Show", ["CHARGE", "DISCHARGE", "HOLD"], default=["CHARGE", "DISCHARGE"])
    view = cards[cards["action"].isin(filt)].copy()
    if "rank" in view.columns:
        view = view.sort_values("rank")

    for ts, row in view.iterrows():
        flags = str(row.get("flags", "") or "")
        flag_html = " ".join(
            f"<span class='badge b-rose'>{f}</span>" for f in flags.split("|") if f
        )
        conf = float(row.get("confidence", 0.5))
        bar = int(conf * 100)
        st.markdown(
            f"""
            <div class='card'>
              <div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start'>
                <div>
                  {action_badge(row['action'])}
                  <span class='mono' style='margin-left:8px'>
                    {pd.Timestamp(ts).strftime('%H:%M')} · {row['mw']:.0f} MW · SOC {row.get('soc_mwh', float('nan')):.0f} MWh
                  </span>
                  <div class='why'>{row.get('why','')}</div>
                </div>
                <div style='text-align:right;min-width:120px'>
                  <div class='mono' style='font-size:18px;font-weight:700'>{fmt_eur(float(row['expected_eur']), 0)}</div>
                  <div style='font-size:11px;color:#8B95A9;margin-top:4px'>confidence {conf:.0%}</div>
                  <div style='height:6px;background:#1C2433;border-radius:99px;margin-top:6px'>
                    <div style='height:6px;width:{bar}%;background:{TEAL};border-radius:99px'></div>
                  </div>
                  <div style='margin-top:8px'>{flag_html}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if view.empty:
        callout("Nothing in the filter. HOLDs are most of the year — that is a desk, not a Kaggle loop.")

    show_cols = [c for c in
                 ["action", "mw", "price_p50", "confidence", "expected_eur", "flags", "why", "rank"]
                 if c in cards.columns]
    st.dataframe(cards[show_cols], width="stretch", height=320)
