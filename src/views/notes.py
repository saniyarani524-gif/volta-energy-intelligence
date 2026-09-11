"""Page 9 — Notes. Assumptions, leakage, how to reproduce."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.features import MIN_LAG_HOURS, PRICE_FEATURES, REALISED_NOW
from src.ui import callout, hero, page_boot, market as load_market
from src.app_data import data_status


def render() -> None:
    page_boot()
    hero(
        "09 · Notes",
        "Every number is from a pipeline that starts at notebook 01",
        "This page is the interview. If a claim cannot be re-run, it does not belong "
        "in the README, the cards, or a recruiter’s notebook.",
    )

    status = data_status()
    st.markdown(
        "<div class='eyebrow'>Local artifacts</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    labels = {
        "market": "01 market.parquet",
        "forecasts_2018": "02 2018 forecasts",
        "walkforward": "02 walk-forward",
        "cards": "04 action cards",
        "dispatch": "03 daily P&L",
        "models": "02 XGB json",
    }
    for i, (k, lab) in enumerate(labels.items()):
        with cols[i % 3]:
            ok = status.get(k, False)
            st.markdown(
                f"<div class='card'><span class='badge b-{'live' if ok else 'rose'}'>"
                f"{'OK' if ok else 'MISSING'}</span>"
                f"<div style='margin-top:8px'>{lab}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='eyebrow'>Assumptions we owe an interviewer</div>", unsafe_allow_html=True)
    assumptions = [
        ("Timezone", "Energy timestamps are CET/CEST already. We convert via UTC → Europe/Madrid. DST 23/25-hour days are first-class in the LP."),
        ("TSO price_da", "Treated as the operator/OMIE day-ahead *forecast* we must beat, not as the auction clearing price. That is the dataset’s own benchmark."),
        ("Weather as NWP", "Hour-t weather stands in for the D-1 numerical weather prediction. We do not have the real NWP vintage."),
        ("City weights", "Madrid 28 / BCN 27 / Valencia 22 / Seville 13 / Bilbao 10. 2015-era metro shares, not official INE weights."),
        ("Gaps", "Holes ≤ 6 h are linearly interpolated. Longer holes stay NaN — we do not invent a missing peak."),
        ("Efficiency", "Round-trip 88% split equally: η_c = η_d = √0.88. Interviews will ask; there is no better split in this dataset."),
        ("Cyclic SOC", "E_T = E_0 each day so we cannot drain the pack on day 1 and call it alpha."),
        ("Throughput", "1.0 cycle/day cap, counted as MWh discharged to the grid. Warranty / degradation proxy."),
        ("Degradation", "€3 / MWh discharged. An assumption, parked in config, not fitted."),
        ("Capex", "€350 / kWh, 2018-era turnkey Li-ion, not 2024 cell prices. 15-year life, 7% WACC, no residual."),
        ("Day-level scale", "Uncertainty and breakeven cut the *whole day*. Hour-level MW edits would break SOC feasibility."),
        ("SPIKE", "A flag, never a silent rewrite. We want the discharge into a €90+ print."),
    ]
    for title, body in assumptions:
        st.markdown(
            f"<div class='card'><div class='eyebrow'>{title}</div>"
            f"<div class='why'>{body}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='eyebrow'>Leakage contract</div>", unsafe_allow_html=True)
    callout(
        f"Anything realised (price, load, generation, net load) may only enter a forecast "
        f"as a lag of ≥ {MIN_LAG_HOURS} h, or as a rolling statistic computed on a series "
        f"that has already been shifted by {MIN_LAG_HOURS} h. Calendar, TSO day-ahead columns "
        f"(price_da, load_da, forecast_solar_da, forecast_wind_da) and weather-as-NWP are legal at t."
    )
    st.caption("Realised-now columns that must never appear in a feature list")
    st.code(", ".join(sorted(REALISED_NOW)), language="text")
    st.caption(f"PRICE_FEATURES ({len(PRICE_FEATURES)} cols)")
    st.code(", ".join(PRICE_FEATURES), language="text")

    st.markdown("<div class='eyebrow'>Config (the numbers the code actually uses)</div>", unsafe_allow_html=True)
    cfg = {
        "BATTERY_POWER_MW": config.BATTERY_POWER_MW,
        "BATTERY_CAPACITY_MWH": config.BATTERY_CAPACITY_MWH,
        "ROUND_TRIP_EFFICIENCY": config.ROUND_TRIP_EFFICIENCY,
        "MIN_SOC_PCT": config.MIN_SOC_PCT,
        "MAX_CYCLES_PER_DAY": config.MAX_CYCLES_PER_DAY,
        "DEGRADATION_COST_EUR_PER_MWH": config.DEGRADATION_COST_EUR_PER_MWH,
        "BATTERY_CAPEX_EUR_PER_KWH": config.BATTERY_CAPEX_EUR_PER_KWH,
        "WACC": config.WACC,
        "ASSET_LIFE_YEARS": config.ASSET_LIFE_YEARS,
        "TRAIN_END": config.TRAIN_END,
        "TEST_START": config.TEST_START,
        "HORIZON_HOURS": config.HORIZON_HOURS,
        "MIN_LAG_HOURS": config.MIN_LAG_HOURS,
        "ACTION_MW_EPS": config.ACTION_MW_EPS,
        "UNCERTAINTY_SOFT": config.UNCERTAINTY_SOFT,
        "UNCERTAINTY_HARD": config.UNCERTAINTY_HARD,
        "SPIKE_EUR": config.SPIKE_EUR,
    }
    st.dataframe(
        pd.Series({k: str(v) for k, v in cfg.items()}, name="value"),
        width="stretch",
    )

    st.markdown("<div class='eyebrow'>Reproduce</div>", unsafe_allow_html=True)
    st.code(
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements-dev.txt\n"
        "jupyter notebook notebooks/01_eda.ipynb\n"
        "jupyter notebook notebooks/02_forecasting.ipynb\n"
        "jupyter notebook notebooks/03_dispatch.ipynb\n"
        "jupyter notebook notebooks/04_decisions.ipynb\n"
        "pytest -q\n"
        "streamlit run app.py",
        language="bash",
    )
    callout(
        "Data: Kaggle <i>Hourly energy demand, generation and weather</i> (nicholasjhana), CC0. "
        "ENTSO-E / Red Eléctrica / OpenWeather, Spain 2015–2018. CSVs stay in data/raw/ and are git-ignored. "
        "Never commit kaggle.json, cache/, models/, or parquet."
    )
    mkt = load_market()
    if mkt is not None:
        st.caption(
            f"Loaded market {len(mkt):,} rows · {mkt.index.min()} → {mkt.index.max()} · "
            f"{mkt.shape[1]} columns"
        )
