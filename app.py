"""VOLTA — Energy Market Intelligence & Battery Trading Desk.

Week 1 build: project shell + live status page.
The full control-room app (forecasting, dispatch, backtest) lands in Weeks 4-10.
Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="VOLTA — Energy Market Intelligence",
                   page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- design system
BG = "#0B0E13"; PANEL = "#10141B"; CARD = "#141924"; BORDER = "#222B3A"
TEXT = "#E7ECF5"; MUTED = "#8B95A9"; FAINT = "#5C6678"
ACCENT = "#2DD4BF"; ACCENT2 = "#F5B84C"  # electric teal + energy amber

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* {{ font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}
html, body, [class*="css"] {{ color: {TEXT}; }}
.stApp {{ background: {BG}; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ background: {PANEL};
    border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

.eyebrow {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .14em;
    color: {ACCENT}; font-weight: 700; }}
.hero {{ background: linear-gradient(160deg, {CARD} 0%, #101724 60%, #0D1620 100%);
    border: 1px solid {BORDER}; border-radius: 18px; padding: 34px 38px;
    margin: 6px 0 18px; position: relative; overflow: hidden; }}
.hero::after {{ content: '⚡'; position: absolute; right: 30px; top: 50%;
    transform: translateY(-50%); font-size: 120px; opacity: .06; }}
.hero h1 {{ font-size: 34px; font-weight: 800; letter-spacing: -.02em;
    margin: 6px 0 10px; }}
.hero p {{ color: {MUTED}; font-size: 15px; max-width: 640px; line-height: 1.6; }}

.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 4px 0 20px; }}
.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 16px 18px; }}
.kpi .label {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em;
    color: {MUTED}; margin-bottom: 6px; white-space: nowrap; }}
.kpi .value {{ font-size: 22px; font-weight: 700; }}
.kpi .sub {{ font-size: 11.5px; color: {FAINT}; margin-top: 4px; }}

.phase {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT};
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 14px; }}
.phase .num {{ font-size: 20px; font-weight: 800; color: {ACCENT}; min-width: 34px; }}
.phase .body {{ flex: 1; }}
.phase .title {{ font-size: 14px; font-weight: 650; }}
.phase .desc {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}

.badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 10.5px; font-weight: 700; letter-spacing: .03em; white-space: nowrap; }}
.b-live  {{ background: rgba(45,212,191,.12); color: {ACCENT};
    border: 1px solid rgba(45,212,191,.35); }}
.b-queue {{ background: rgba(139,149,169,.12); color: {MUTED};
    border: 1px solid rgba(139,149,169,.3); }}
.b-amber {{ background: rgba(245,184,76,.13); color: {ACCENT2};
    border: 1px solid rgba(245,184,76,.35); }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"""
    <div style='display:flex;gap:10px;align-items:center;padding:6px 2px 14px'>
      <div style='font-size:26px'>⚡</div>
      <div>
        <div style='font-size:18px;font-weight:800;letter-spacing:-.01em'>VOLTA</div>
        <div class='eyebrow'>Energy Market Intelligence</div>
      </div>
    </div>
    <div class='badge b-live'>REAL DATA · ENTSO-E / REE · SPAIN</div>
    <p style='font-size:12px;color:{MUTED};line-height:1.6;margin-top:14px'>
    Turning four years of hourly market prices, weather and renewables into
    smarter battery-trading decisions.</p>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='border-top:1px solid {BORDER};padding-top:12px;margin-top:6px;
         font-size:11.5px;color:{FAINT};line-height:2'>
      <b style='color:{MUTED}'>Market</b> · Spain (MIBEL)<br>
      <b style='color:{MUTED}'>Window</b> · 2015 – 2018 · hourly<br>
      <b style='color:{MUTED}'>Asset</b> · 100 MW / 200 MWh BESS<br>
      <b style='color:{MUTED}'>Build</b> · Week 1 of 12
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='badge b-amber'>Status · Data acquisition ✓</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class='hero'>
  <div class='eyebrow'>Portfolio build · in progress</div>
  <h1>⚡ VOLTA — Energy Market Intelligence<br>&amp; Battery Trading Desk</h1>
  <p>Turn market prices, weather and renewables into smarter trading decisions.
  VOLTA ingests <b style='color:{TEXT}'>real Spanish electricity market data</b>
  (ENTSO-E / Red Eléctrica), forecasts day-ahead prices against the grid operator's
  own forecast, and optimises a 100&nbsp;MW / 200&nbsp;MWh battery's charge-discharge
  schedule — every recommendation shipped with a transparent <i>why</i>.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- KPIs
st.markdown("""
<div class='kpi-grid'>
  <div class='kpi'><div class='label'>Dataset</div><div class='value'>35,064</div>
    <div class='sub'>hourly market rows · 2015–2018</div></div>
  <div class='kpi'><div class='label'>Weather</div><div class='value'>178,396</div>
    <div class='sub'>observations · 5 Spanish cities</div></div>
  <div class='kpi'><div class='label'>Sources</div><div class='value'>ENTSO-E + REE</div>
    <div class='sub'>real TSO &amp; market data · CC0</div></div>
  <div class='kpi'><div class='label'>Target claim</div><div class='value'>Beat the TSO</div>
    <div class='sub'>our forecast vs grid operator's own</div></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- roadmap
left, right = st.columns([1.5, 1])

with left:
    st.markdown(f"<div class='eyebrow' style='margin:6px 0 10px'>Build roadmap · 12 weeks</div>",
                unsafe_allow_html=True)
    for num, title, desc, badge in [
        ("01", "Foundations — data pipeline & market analytics",
         "Clean hourly frame · duck curve · merit order · spike taxonomy", "live"),
        ("02", "Forecasting — beat the TSO",
         "XGBoost day-ahead price & load vs operator benchmark · quantile bands", "queue"),
        ("03", "Battery optimizer — the money module",
         "LP dispatch · rolling horizon · 3-year € P&L backtest · degradation", "queue"),
        ("04", "Product — control-room UI & case study",
         "9-page Streamlit app · decision engine · HTML case study · launch", "queue"),
    ]:
        badge_html = ("<span class='badge b-live'>IN PROGRESS</span>" if badge == "live"
                      else "<span class='badge b-queue'>QUEUED</span>")
        st.markdown(f"""
        <div class='phase'>
          <div class='num'>{num}</div>
          <div class='body'><div class='title'>{title}</div>
          <div class='desc'>{desc}</div></div>
          <div>{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown(f"<div class='eyebrow' style='margin:6px 0 10px'>The asset</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kpi'><div class='label'>Grid-scale battery (BESS)</div>
      <div class='value'>100 MW / 200 MWh</div>
      <div class='sub'>2-hour duration · 88% round-trip efficiency · 1 cycle/day cap</div>
      <p style='font-size:12px;color:{MUTED};line-height:1.7;margin-top:10px'>
      Buy low at night &amp; solar noon — sell into the evening peak.
      The optimizer turns price spreads into revenue while respecting
      degradation and safety limits.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kpi' style='margin-top:12px'><div class='label'>Why it matters</div>
      <p style='font-size:12px;color:{MUTED};line-height:1.7;margin:4px 0 0'>
      Battery storage is the backbone of the renewable transition — and
      price arbitrage is its core trading strategy. This desk simulates it
      on real market data.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("Synthetic-free zone: every number here comes from public ENTSO-E / Red Eléctrica "
           "data via the Kaggle open dataset (CC0). Built by Saniya · Week 1 of 12.")
