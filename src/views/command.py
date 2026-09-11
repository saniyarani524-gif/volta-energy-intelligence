"""Page 1 — Command. The desk at a glance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.decisions import decide_day
from src.ui import (
    AMBER,
    CHARGE,
    DISCHARGE,
    action_badge,
    apply_layout,
    callout,
    chart,
    cards as load_cards,
    dispatch_daily,
    doctor_box,
    forecasts_2018,
    fmt_eur,
    fmt_eur_k,
    hero,
    kpis,
    market as load_market,
    page_boot,
    tape,
)
from src.app_data import slice_day


def _day_cards(day):
    cached = load_cards()
    if cached is not None and not cached.empty:
        piece = slice_day(cached, day)
        if not piece.empty:
            return piece
    fc = forecasts_2018()
    if fc is not None:
        piece = slice_day(fc, day)
        if not piece.empty and "price_p50" in piece.columns:
            d = decide_day(piece["price_p50"], piece.get("price_p10"), piece.get("price_p90"))
            return d.cards
    mkt = load_market()
    if mkt is None:
        return None
    piece = slice_day(mkt, day)
    if piece.empty:
        return None
    d = decide_day(piece["price_actual"])
    return d.cards


def render() -> None:
    day = page_boot()
    mkt = tape()
    if mkt is None:
        hero(
            "01 · Command",
            "⚡ VOLTA trading desk",
            "The tape is not on disk yet. The doctor below says exactly why.",
        )
        doctor_box()
        return
    if load_market() is None:
        callout(
            "Full 2015–18 <span class='mono'>market.parquet</span> not found. "
            "Showing the 2018 forecast cache so the desk still opens. Pulse / Stack need the full tape.",
            "warn",
        )
        doctor_box()
    day_mkt = slice_day(mkt, day)

    hero(
        "01 · Command",
        "⚡ VOLTA trading desk",
        "Replay of the Spanish day-ahead. Forecasts beat the TSO, an LP "
        "sizes a 100 MW / 200 MWh battery, and every hour ships with a why. "
        "This is not a live feed — the tape ends 31 Dec 2018.",
    )

    day_cards = _day_cards(day)
    daily = dispatch_daily()
    fc = forecasts_2018()

    expected = 0.0
    scale = 1.0
    n_money = 0
    top_action = "HOLD"
    if day_cards is not None and not day_cards.empty:
        expected = float(day_cards["expected_eur"].sum()) if "expected_eur" in day_cards else 0.0
        if "scale" in day_cards:
            scale = float(day_cards["scale"].iloc[0])
        n_money = int((day_cards["action"] != "HOLD").sum())
        money = day_cards[day_cards["action"] != "HOLD"]
        if not money.empty:
            top = money.reindex(money["expected_eur"].abs().sort_values(ascending=False).index).iloc[0]
            top_action = str(top["action"])

    volta_2018 = None
    if daily is not None:
        v = daily[daily["policy"] == "volta"]
        if not v.empty:
            v2018 = v.loc[v.index.year == 2018] if hasattr(v.index, "year") else v
            volta_2018 = float(v2018["profit_eur"].sum()) if len(v2018) else None

    mae_line = "run notebook 02"
    if fc is not None:
        mae_tso = float((fc["price_da"] - fc["price_actual"]).abs().mean())
        mae_us = float((fc["price_pred"] - fc["price_actual"]).abs().mean())
        mae_line = f"€{mae_us:.2f} vs TSO €{mae_tso:.2f}"

    kpis(
        [
            (day.isoformat(), top_action, f"{n_money} money hours · scale {scale:.0%}"),
            ("Day expected", fmt_eur(expected, 0), "P50 × MW − degradation (forecast, not settled)"),
            ("2018 VOLTA P&L", fmt_eur_k(volta_2018) if volta_2018 is not None else "—", "settled on realised spot"),
            ("2018 price MAE", mae_line, "walk-forward / XGB vs operator"),
        ]
    )

    if day_mkt.empty:
        callout(f"No market rows for {day.isoformat()}. DST or out-of-range.", "warn")
        return

    left, right = st.columns([1.35, 1])

    with left:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=day_mkt.index,
                y=day_mkt["price_actual"],
                name="spot",
                line=dict(color=AMBER, width=2.2),
            ),
            secondary_y=False,
        )
        if "price_da" in day_mkt:
            fig.add_trace(
                go.Scatter(
                    x=day_mkt.index,
                    y=day_mkt["price_da"],
                    name="TSO DA",
                    line=dict(color="#64748B", width=1.2, dash="dot"),
                ),
                secondary_y=False,
            )
        if day_cards is not None and not day_cards.empty:
            fig.add_trace(
                go.Bar(
                    x=day_cards.index,
                    y=day_cards["mw"].where(day_cards["action"] == "CHARGE", 0),
                    name="charge",
                    marker_color=CHARGE,
                    opacity=0.55,
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Bar(
                    x=day_cards.index,
                    y=-day_cards["mw"].where(day_cards["action"] == "DISCHARGE", 0),
                    name="discharge",
                    marker_color=DISCHARGE,
                    opacity=0.7,
                ),
                secondary_y=True,
            )
        fig.update_yaxes(title_text="€/MWh", secondary_y=False)
        fig.update_yaxes(title_text="MW  (+ charge / − discharge)", secondary_y=True, showgrid=False)
        apply_layout(fig, height=380, barmode="relative", title=f"{day.isoformat()} · price vs action")
        chart(fig)

    with right:
        st.markdown("<div class='eyebrow'>Top cards · ranked by |€|</div>", unsafe_allow_html=True)
        if day_cards is None or day_cards.empty:
            callout("No cards for this day. Run notebook 04, or pick a 2018 date.", "warn")
        else:
            ranked = day_cards.sort_values("rank") if "rank" in day_cards.columns else day_cards
            shown = 0
            for ts, row in ranked.iterrows():
                if row["action"] == "HOLD" and shown >= 3:
                    continue
                if shown >= 5:
                    break
                flags = str(row.get("flags", "") or "")
                flag_html = ""
                if flags:
                    flag_html = " ".join(
                        f"<span class='badge b-rose'>{f}</span>" for f in flags.split("|") if f
                    )
                st.markdown(
                    f"""
                    <div class='card'>
                      <div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>
                        <div>{action_badge(row['action'])}
                          <span class='mono' style='margin-left:8px;font-size:13px'>
                            {pd.Timestamp(ts).strftime('%H:%M')} · {row['mw']:.0f} MW
                          </span>
                        </div>
                        <div class='mono'>{fmt_eur(float(row['expected_eur']), 0)}</div>
                      </div>
                      <div class='why'>{row.get('why', '')}</div>
                      <div style='margin-top:8px'>{flag_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                shown += 1
            if shown == 0:
                st.markdown(
                    "<div class='card'><div class='why'>All HOLD. Spread does not clear round-trip + degradation, "
                    "or the LP has no flow.</div></div>",
                    unsafe_allow_html=True,
                )

    callout(
        "<b>How to read this.</b> Expected € is the P50 cashflow of the scaled schedule — it is not "
        "settled P&amp;L. Settled money lives on Ledger. Scale is day-level so SOC physics stay feasible. "
        "A SPIKE flag is a warning, not a silent rewrite."
    )
