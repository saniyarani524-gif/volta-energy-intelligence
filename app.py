"""VOLTA — Energy Market Intelligence & Battery Trading Desk.

Nine-page control room. Numbers come from notebooks 01–04 (cache / parquet).
Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="VOLTA — Trading Desk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.views import cards, command, dispatch, forecast, ledger, market, notes, risk, stack  # noqa: E402

pages = {
    "Desk": [
        st.Page(command.render, title="Command", icon="⚡", default=True, url_path="command"),
        st.Page(cards.render, title="Cards", icon="🗂️", url_path="cards"),
        st.Page(dispatch.render, title="Dispatch", icon="🔋", url_path="dispatch"),
    ],
    "Market": [
        st.Page(market.render, title="Pulse", icon="📈", url_path="pulse"),
        st.Page(stack.render, title="Stack", icon="🏭", url_path="stack"),
        st.Page(forecast.render, title="Forecast", icon="🎯", url_path="forecast"),
    ],
    "Results": [
        st.Page(ledger.render, title="Ledger", icon="📒", url_path="ledger"),
        st.Page(risk.render, title="Risk", icon="⚠️", url_path="risk"),
        st.Page(notes.render, title="Notes", icon="📝", url_path="notes"),
    ],
}

st.navigation(pages, position="sidebar").run()
