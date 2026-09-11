"""VOLTA control-room design system.

Teal + amber on near-black. Shared by every page so the desk does not
drift into nine different apps.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_data
from src.config import (
    BATTERY_CAPACITY_MWH,
    BATTERY_POWER_MW,
    COMPANY,
    ROUND_TRIP_EFFICIENCY,
    TAGLINE,
)
from src.market_io import market_parquet, searched_market_paths

BG = "#0B0E13"
PANEL = "#10141B"
CARD = "#141924"
BORDER = "#222B3A"
TEXT = "#E7ECF5"
MUTED = "#8B95A9"
FAINT = "#5C6678"
TEAL = "#2DD4BF"
AMBER = "#F5B84C"
ROSE = "#FB7185"
BLUE = "#60A5FA"
VIOLET = "#A78BFA"
GREEN = "#34D399"

CHARGE = TEAL
DISCHARGE = AMBER
HOLD = FAINT

MIX_COLORS = {
    "nuclear": "#A78BFA",
    "wind": "#38BDF8",
    "hydro": "#2563EB",
    "gas": "#F97316",
    "coal": "#78716C",
    "solar": "#EAB308",
    "other": "#86EFAC",
}

ACTION_COLORS = {"CHARGE": CHARGE, "DISCHARGE": DISCHARGE, "HOLD": HOLD}


def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, label {{
  font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: {BG}; color: {TEXT}; }}
#MainMenu, header[data-testid="stHeader"], footer, .stDeployButton {{
  visibility: hidden; height: 0;
}}
section[data-testid="stSidebar"] {{
  background: {PANEL};
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
div[data-testid="stToolbar"] {{ display: none; }}

.eyebrow {{
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .16em;
  color: {TEAL}; font-weight: 700; margin-bottom: 4px;
}}
.hero {{
  background: linear-gradient(160deg, {CARD} 0%, #101724 60%, #0D1620 100%);
  border: 1px solid {BORDER}; border-radius: 16px; padding: 26px 30px;
  margin: 0 0 16px; position: relative; overflow: hidden;
}}
.hero h1 {{
  font-size: 28px; font-weight: 700; letter-spacing: -.02em;
  margin: 4px 0 8px; line-height: 1.15;
}}
.hero p {{ color: {MUTED}; font-size: 14px; max-width: 720px; line-height: 1.55; margin: 0; }}

.kpi-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px; margin: 0 0 16px;
}}
.kpi {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 14px 16px;
}}
.kpi .label {{
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em;
  color: {MUTED}; margin-bottom: 6px;
}}
.kpi .value {{ font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }}
.kpi .sub {{ font-size: 11.5px; color: {FAINT}; margin-top: 4px; }}

.badge {{
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 10.5px; font-weight: 700; letter-spacing: .04em; white-space: nowrap;
}}
.b-live {{ background: rgba(45,212,191,.12); color: {TEAL}; border: 1px solid rgba(45,212,191,.35); }}
.b-amber {{ background: rgba(245,184,76,.13); color: {AMBER}; border: 1px solid rgba(245,184,76,.35); }}
.b-rose {{ background: rgba(251,113,133,.13); color: {ROSE}; border: 1px solid rgba(251,113,133,.35); }}
.b-mute {{ background: rgba(139,149,169,.12); color: {MUTED}; border: 1px solid rgba(139,149,169,.3); }}
.b-charge {{ background: rgba(45,212,191,.14); color: {TEAL}; border: 1px solid rgba(45,212,191,.4); }}
.b-discharge {{ background: rgba(245,184,76,.14); color: {AMBER}; border: 1px solid rgba(245,184,76,.4); }}
.b-hold {{ background: rgba(92,102,120,.16); color: {MUTED}; border: 1px solid rgba(92,102,120,.35); }}

.callout {{
  background: {CARD}; border: 1px solid {BORDER}; border-left: 3px solid {TEAL};
  border-radius: 10px; padding: 12px 16px; color: {MUTED}; font-size: 13.5px;
  line-height: 1.6; margin: 8px 0 16px;
}}
.callout.warn {{ border-left-color: {AMBER}; }}
.callout.danger {{ border-left-color: {ROSE}; }}

.card {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 14px 16px; margin-bottom: 10px;
}}
.why {{
  font-size: 13px; color: {TEXT}; line-height: 1.55; margin-top: 8px;
  font-family: 'IBM Plex Sans', sans-serif;
}}
.mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }}

div[data-testid="stSidebarNav"] {{ padding-top: 4px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
</style>
""",
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{eyebrow}</div>"
        f"<h1>{title}</h1><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def kpis(items: list[tuple[str, str, str]]) -> None:
    cells = []
    for label, value, sub in items:
        cells.append(
            f"<div class='kpi'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"<div class='sub'>{sub}</div></div>"
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    cls = "callout" + ("" if kind == "info" else f" {kind}")
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "live") -> str:
    return f"<span class='badge b-{kind}'>{text}</span>"


def action_badge(action: str) -> str:
    kind = {"CHARGE": "charge", "DISCHARGE": "discharge", "HOLD": "hold"}.get(action, "mute")
    return badge(action, kind)


def fmt_eur(x: float, digits: int = 0) -> str:
    if x is None or not pd.notna(x):
        return "—"
    sign = "−" if x < 0 else ""
    return f"{sign}€{abs(x):,.{digits}f}"


def fmt_eur_k(x: float) -> str:
    if x is None or not pd.notna(x):
        return "—"
    ax = abs(x)
    sign = "−" if x < 0 else ""
    if ax >= 1_000_000:
        return f"{sign}€{ax/1_000_000:.1f}m"
    if ax >= 1_000:
        return f"{sign}€{ax/1_000:.0f}k"
    return f"{sign}€{ax:.0f}"


def fmt_mw(x: float) -> str:
    if x is None or not pd.notna(x):
        return "—"
    return f"{x:,.0f} MW"


def apply_layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="IBM Plex Sans, sans-serif", size=12),
        margin=dict(l=48, r=16, t=36, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis=dict(gridcolor="#1C2433", zerolinecolor="#1C2433", linecolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor="#1C2433", zerolinecolor="#1C2433", linecolor=BORDER, showgrid=True),
        colorway=[TEAL, AMBER, ROSE, BLUE, VIOLET, "#94A3B8"],
        **kwargs,
    )
    return fig


def chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:
        st.exception(exc)


def table(df, **kwargs) -> None:
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except Exception:
        st.dataframe(df, **kwargs)


def market() -> pd.DataFrame | None:
    # Do not cache None — a first run before unzip would poison the session.
    return app_data.read_market()


def forecasts_2018() -> pd.DataFrame | None:
    return app_data.read_forecasts_2018()


def walkforward() -> pd.DataFrame | None:
    return app_data.read_walkforward()


def cards() -> pd.DataFrame | None:
    return app_data.read_cards()


def dispatch_daily() -> pd.DataFrame | None:
    return app_data.read_dispatch_daily()


def dispatch_hourly_perfect() -> pd.DataFrame | None:
    return app_data.read_dispatch_hourly_perfect()


def importance() -> pd.Series | None:
    return app_data.feature_importance()


def tape() -> pd.DataFrame | None:
    """Full 2015–18 market, else the 2018 forecast cache so Command still opens."""
    m = market()
    if m is not None and not m.empty:
        return m
    fc = forecasts_2018()
    if fc is not None and not fc.empty:
        return fc
    return None


def doctor_box() -> None:
    from src.app_data import READ_ERRORS
    from src.doctor import report

    empty = market_parquet() is None and forecasts_2018() is None
    with st.expander("Desk doctor · why a page is empty", expanded=empty or bool(READ_ERRORS)):
        st.code(report(), language="text")
        status = app_data.data_status()
        if status.get("raw_csvs") and market_parquet() is None:
            if st.button("Build market.parquet from data/raw", type="primary"):
                with st.spinner("Cleaning 35,064 hours · one-time pass…"):
                    out = app_data.try_build_market()
                st.success(f"Wrote {out}")
                st.rerun()
        st.caption("In the repo:  python -m src.doctor")


def need(df: pd.DataFrame | None, what: str, notebook: str) -> pd.DataFrame:
    if df is not None and not df.empty:
        return df
    callout(
        f"Could not load <b>{what}</b>. Open <i>Desk doctor</i> — usually Streamlit "
        f"was started outside the repo, or <span class='mono'>pyarrow</span> is missing.",
        "warn",
    )
    doctor_box()
    st.stop()
    raise RuntimeError("unreachable")


def desk_day() -> date:
    """Sidebar date that every page shares."""
    mkt = tape()
    span = app_data.available_days(mkt)
    fallback = app_data.DEFAULT_DAY
    if span is not None:
        fallback = app_data.clamp_day(fallback, span)
    if "desk_day" not in st.session_state:
        st.session_state.desk_day = fallback

    with st.sidebar:
        st.markdown(
            f"""
            <div style='display:flex;gap:10px;align-items:center;padding:4px 2px 12px'>
              <div style='font-size:26px'>⚡</div>
              <div>
                <div style='font-size:18px;font-weight:700;letter-spacing:-.01em'>VOLTA</div>
                <div class='eyebrow' style='margin:0'>Energy Market Intelligence</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(badge("REPLAY DESK · SPAIN 2015–2018", "live"), unsafe_allow_html=True)
        st.caption(TAGLINE)

        lo = span[0] if span else date(2015, 1, 1)
        hi = span[1] if span else date(2018, 12, 31)
        st.caption("Story days")
        bcols = st.columns(2)
        for i, (d, name, _) in enumerate(app_data.PRESET_DAYS):
            if bcols[i % 2].button(name, key=f"preset_{i}", use_container_width=True):
                st.session_state.desk_day = app_data.clamp_day(d, span)
                st.rerun()
        st.date_input(
            "Madrid-local day",
            key="desk_day",
            min_value=lo,
            max_value=hi,
        )

        st.markdown(
            f"""
            <div style='border-top:1px solid {BORDER};padding-top:12px;margin-top:8px;
                 font-size:11.5px;color:{FAINT};line-height:1.9'>
              <b style='color:{MUTED}'>{COMPANY}</b><br>
              Asset · {BATTERY_POWER_MW:.0f} MW / {BATTERY_CAPACITY_MWH:.0f} MWh<br>
              Round-trip · {ROUND_TRIP_EFFICIENCY:.0%} · 1 cycle/day<br>
              Window · ENTSO-E / REE · CC0
            </div>
            """,
            unsafe_allow_html=True,
        )
        status = app_data.data_status()
        bits = " · ".join(
            f"{k.replace('_', ' ')} {'✓' if v else '✗'}" for k, v in status.items()
        )
        st.caption(bits)
        mp = market_parquet()
        st.caption("market → " + (str(mp) if mp else "NOT FOUND"))

    day = st.session_state.desk_day
    if hasattr(day, "date"):
        day = day.date()
    return app_data.clamp_day(day, span)


def page_boot() -> date:
    inject_css()
    return desk_day()
