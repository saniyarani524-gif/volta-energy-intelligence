"""Page 3 — Stack. Duck curve, merit order, spikes, mix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import generation_mix
from src.ui import (
    AMBER,
    MIX_COLORS,
    ROSE,
    TEAL,
    apply_layout,
    callout,
    chart,
    hero,
    kpis,
    market as load_market,
    need,
    page_boot,
)


def render() -> None:
    page_boot()
    mkt = need(load_market(), "market.parquet", "notebooks/01_eda.ipynb")

    hero(
        "03 · Stack",
        "Duck, merit order, spikes",
        "The 2015–18 “transition” on the peninsula is coal → gas, not fossil → solar. "
        "Hydro is the wild card. Spikes are a January-2017 gas-on-the-margin story, "
        "not a low-wind story.",
    )

    mix = generation_mix(mkt)
    twh = mix.groupby(mix.index.year).sum() / 1e6
    coal_15, coal_18 = float(twh.loc[2015, "coal"]), float(twh.loc[2018, "coal"])
    gas_15, gas_18 = float(twh.loc[2015, "gas"]), float(twh.loc[2018, "gas"])
    kpis(
        [
            ("Coal 2015 → 2018", f"{coal_15:.0f} → {coal_18:.0f} TWh", "structurally retiring"),
            ("Gas 2015 → 2018", f"{gas_15:.0f} → {gas_18:.0f} TWh", "CCGT absorbs the gap"),
            ("Hydro range", f"{twh['hydro'].min():.0f}–{twh['hydro'].max():.0f} TWh", "2017 drought vs 2016 wet"),
            ("Solar", "flat", "policy freeze — night output is CSP, not a bug"),
        ]
    )

    tab_duck, tab_mix, tab_merit, tab_spike = st.tabs(
        ["Duck curve", "Generation mix", "Merit order", "Spike anatomy"]
    )

    with tab_duck:
        net = mkt["net_load"]
        fig = go.Figure()
        palette = ["#64748B", TEAL, AMBER, ROSE]
        for i, year in enumerate(sorted(mkt.index.year.unique())):
            s = net[net.index.year == year]
            by_h = s.groupby(s.index.hour).mean() / 1000
            fig.add_trace(
                go.Scatter(
                    x=list(range(24)), y=by_h.values, name=str(year),
                    line=dict(color=palette[i % 4], width=2.2),
                )
            )
        apply_layout(fig, height=400, title="Net load by hour (GW) — belly at solar noon, ramp into 21:00")
        fig.update_xaxes(title="hour", dtick=2)
        fig.update_yaxes(title="GW")
        chart(fig)
        callout(
            "Intra-day, yes: solar carves a belly 10:00–16:00 and the evening ramp is the "
            "expensive print. Across 2015–18 the duck does <b>not</b> deepen — solar capacity "
            "is frozen. The battery’s edge here is the intra-day spread, not a growing duck."
        )

    with tab_mix:
        monthly = mix.resample("MS").mean() / 1000
        fig = go.Figure()
        for col in monthly.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly.index, y=monthly[col], name=col,
                    stackgroup="mix", line=dict(width=0),
                    fillcolor=MIX_COLORS.get(col),
                    hovertemplate=f"{col} %{{y:.1f}} GW<extra></extra>",
                )
            )
        apply_layout(fig, height=420, title="Monthly generation mix (GW) — coal gives way to gas")
        fig.update_yaxes(title="GW", range=[0, 32])
        chart(fig)
        st.dataframe(twh.round(1), use_container_width=True)
        callout(
            "A stack that is more CCGT-heavy is <b>more</b> marginal-price volatile, not less. "
            "Gas-on-the-margin hours are exactly the hours with a fat intra-day spread. "
            "The asset gets more valuable as coal retires, even without more solar."
        )

    with tab_merit:
        sample = mkt[["price_actual", "net_load", "gen_wind", "load_actual"]].dropna()
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=7)
        fig = go.Figure(
            go.Scattergl(
                x=sample["net_load"] / 1000,
                y=sample["price_actual"],
                mode="markers",
                marker=dict(size=4, color=sample["gen_wind"], colorscale="Tealgrn",
                            colorbar=dict(title="wind MW"), opacity=0.55),
                name="hours",
                hovertemplate="net load %{x:.1f} GW<br>€%{y:.1f}<extra></extra>",
            )
        )
        apply_layout(fig, height=420, title="Price vs residual demand — colour is wind")
        fig.update_xaxes(title="net load GW")
        fig.update_yaxes(title="€/MWh")
        chart(fig)
        clean = mkt.dropna(subset=["price_actual", "net_load", "load_actual", "gen_wind"])
        corr = {
            "net_load": clean["net_load"].corr(clean["price_actual"]),
            "load": clean["load_actual"].corr(clean["price_actual"]),
            "wind": clean["gen_wind"].corr(clean["price_actual"]),
            "solar": clean["gen_solar"].corr(clean["price_actual"]),
        }
        callout(
            f"Residual demand correlates <b>{corr['net_load']:.2f}</b> with price, ahead of raw load "
            f"({corr['load']:.2f}). Wind is the depressant ({corr['wind']:.2f}). "
            "That is the merit-order story in one line."
        )

    with tab_spike:
        top = mkt.nlargest(50, "price_actual")[
            ["price_actual", "load_actual", "gen_wind", "gen_gas", "gen_coal", "net_load"]
        ].copy()
        top["when"] = top.index.strftime("%Y-%m-%d %H:%M")
        top["dow"] = top.index.day_name()
        years = top.index.year.value_counts().to_dict()
        weekend = int((top.index.dayofweek >= 5).sum())
        kpis(
            [
                ("Top-50 hours", "all in 2017" if years.get(2017, 0) == 50 else str(years),
                 "not a four-year phenomenon"),
                ("Weekends in top 50", str(weekend), "weekday peakers"),
                ("Mean wind on spikes", f"{top['gen_wind'].mean():,.0f} MW", "this is not a low-wind story"),
                ("Mean gas on spikes", f"{top['gen_gas'].mean():,.0f} MW", "CCGT on the margin"),
            ]
        )
        show = top[["when", "dow", "price_actual", "gen_wind", "gen_gas", "load_actual"]].head(20)
        show = show.rename(columns={
            "price_actual": "€/MWh", "gen_wind": "wind MW", "gen_gas": "gas MW", "load_actual": "load MW",
        })
        st.dataframe(show.set_index("when").round(1), use_container_width=True)
        callout(
            "January 2017, weekdays, gas running hard. A 2-hour battery that can charge overnight "
            "and hit 19:00–21:00 is exactly the shape of these hours. Phase 4 flags them SPIKE "
            "so a trader can agree or override — we do not silently zero a discharge into a print we wanted.",
            "warn",
        )
