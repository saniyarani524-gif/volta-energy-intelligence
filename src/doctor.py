"""Why the desk cannot see the tape. Run:  python -m src.doctor"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from src.market_io import (
    candidate_roots,
    energy_csv,
    find_file,
    market_parquet,
    weather_csv,
)
from src.app_data import READ_ERRORS, read_cards, read_forecasts_2018, read_market


def report() -> str:
    lines = []
    lines.append(f"python     : {sys.executable}")
    lines.append(f"version    : {sys.version.split()[0]}")
    lines.append(f"cwd        : {Path.cwd()}")
    lines.append(f"VOLTA_ROOT : {os.environ.get('VOLTA_ROOT', '(unset)')}")
    try:
        import pyarrow
        lines.append(f"pyarrow    : {pyarrow.__version__}")
    except Exception as e:
        lines.append(f"pyarrow    : MISSING ({e})")
    try:
        import streamlit as st
        lines.append(f"streamlit  : {st.__version__}")
    except Exception as e:
        lines.append(f"streamlit  : MISSING ({e})")
    try:
        import plotly
        lines.append(f"plotly     : {plotly.__version__}")
    except Exception as e:
        lines.append(f"plotly     : MISSING ({e})")

    mp = market_parquet()
    lines.append(f"market     : {mp if mp else 'NOT FOUND'}")
    fc = find_file("cache", "test_2018_forecasts.parquet")
    lines.append(f"forecasts  : {fc if fc else 'NOT FOUND'}")
    cards = find_file("cache", "action_cards_2018.parquet")
    lines.append(f"cards      : {cards if cards else 'NOT FOUND'}")
    lines.append(f"energy csv : {energy_csv() or 'NOT FOUND'}")
    lines.append(f"weather csv: {weather_csv() or 'NOT FOUND'}")

    m = read_market()
    lines.append(f"read market: {None if m is None else m.shape}")
    f = read_forecasts_2018()
    lines.append(f"read 2018  : {None if f is None else f.shape}")
    c = read_cards()
    lines.append(f"read cards : {None if c is None else c.shape}")
    if READ_ERRORS:
        lines.append("read errors:")
        for k, v in READ_ERRORS.items():
            lines.append(f"  {k}")
            lines.append(f"    {v}")
    lines.append("roots tried:")
    for r in candidate_roots()[:10]:
        lines.append(f"  {r}")
    return "\n".join(lines)


def main() -> None:
    print(report())
    m = read_market()
    f = read_forecasts_2018()
    if m is None and f is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
