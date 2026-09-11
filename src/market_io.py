"""Find or rebuild the tidy hourly frame the desk runs on.

``market.parquet`` is git-ignored (it is data). The Streamlit process must
locate it under the *repo* that has ``data/``, not whatever folder the user
happened to ``streamlit run`` from. If the parquet is missing but the two
Kaggle CSVs are present, we rebuild with the same rules as notebook 01.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CITY_WEIGHTS, TZ

DEAD_COLS = [
    "generation hydro pumped storage aggregated",
    "forecast wind offshore eday ahead",
    "generation fossil coal-derived gas",
    "generation fossil oil shale",
    "generation fossil peat",
    "generation geothermal",
    "generation marine",
    "generation wind offshore",
]

RENAME = {
    "price actual": "price_actual",
    "price day ahead": "price_da",
    "total load actual": "load_actual",
    "total load forecast": "load_da",
    "generation biomass": "gen_biomass",
    "generation fossil brown coal/lignite": "gen_lignite",
    "generation fossil gas": "gen_gas",
    "generation fossil hard coal": "gen_coal",
    "generation fossil oil": "gen_oil",
    "generation hydro pumped storage consumption": "gen_hydro_pump",
    "generation hydro run-of-river and poundage": "gen_hydro_ror",
    "generation hydro water reservoir": "gen_hydro_res",
    "generation nuclear": "gen_nuclear",
    "generation other": "gen_other",
    "generation other renewable": "gen_other_re",
    "generation solar": "gen_solar",
    "generation waste": "gen_waste",
    "generation wind onshore": "gen_wind",
    "forecast solar day ahead": "forecast_solar_da",
    "forecast wind onshore day ahead": "forecast_wind_da",
}

MAX_GAP = 6
CACHE_FILES = (
    "test_2018_forecasts.parquet",
    "wf_forecasts_2016_2018.parquet",
    "action_cards_2018.parquet",
    "dispatch_perfect_daily.parquet",
    "dispatch_volta_daily.parquet",
    "dispatch_tso_daily.parquet",
    "dispatch_hourly_perfect.parquet",
)


def _looks_like_root(p: Path) -> bool:
    return (p / "src" / "config.py").exists() or (p / "app.py").exists()


def candidate_roots() -> list[Path]:
    """Repo-shaped folders, starting from cwd and this file, plus VOLTA_ROOT."""
    found: list[Path] = []

    def add(p: Path | None) -> None:
        if p is None:
            return
        try:
            p = p.resolve()
        except OSError:
            return
        if p not in found:
            found.append(p)

    env = os.environ.get("VOLTA_ROOT")
    if env:
        add(Path(env))

    here = Path(__file__).resolve().parents[1]
    add(here)
    try:
        add(Path.cwd())
    except OSError:
        pass

    home = Path.home()
    for extra in (
        home / "Documents" / "volta-energy-intelligence",
        home / "Documents" / "GitHub" / "volta-energy-intelligence",
        home / "Desktop" / "volta-energy-intelligence",
        home / "Downloads",
        home / "Downloads" / "volta-energy-intelligence",
        home / "Downloads" / "volta-phase5",
    ):
        add(extra)
        add(extra / "data" / "processed")  # harmless if missing
    # unzip-in-Downloads often yields Downloads/data/processed/market.parquet
    add(home / "Downloads" / "data" / "processed")

    extras: list[Path] = []
    for start in list(found):
        extras.extend([start, *start.parents])
    for p in extras:
        add(p)

    # Prefer folders that actually contain the parquet or the CSVs.
    def score(p: Path) -> tuple[int, int]:
        hits = 0
        if (p / "data" / "processed" / "market.parquet").exists():
            hits += 4
        if (p / "cache" / "test_2018_forecasts.parquet").exists():
            hits += 2
        if (p / "data" / "raw" / "energy_dataset.csv").exists():
            hits += 1
        if _looks_like_root(p):
            hits += 1
        return (-hits, len(str(p)))

    return sorted(found, key=score)


def find_file(*parts: str) -> Path | None:
    for root in candidate_roots():
        path = root.joinpath(*parts)
        if path.is_file():
            return path
        # also allow root itself already being the folder that holds the file
        alt = root.joinpath(parts[-1]) if parts else None
        if alt is not None and alt.is_file() and alt.name == parts[-1]:
            return alt
    return None


def market_parquet() -> Path | None:
    here = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    env = os.environ.get("VOLTA_ROOT")
    pinned = []
    if env:
        pinned.append(Path(env))
    pinned.extend([here, cwd])
    for root in pinned:
        for p in (
            root / "data" / "processed" / "market.parquet",
            root / "market.parquet",
        ):
            if p.is_file():
                return p
    return find_file("data", "processed", "market.parquet")


def cache_dir() -> Path | None:
    for root in candidate_roots():
        d = root / "cache"
        if d.is_dir():
            return d
    # last resort: next to this repo's src/
    p = Path(__file__).resolve().parents[1] / "cache"
    return p if p.exists() else None


def energy_csv() -> Path | None:
    return find_file("data", "raw", "energy_dataset.csv")


def weather_csv() -> Path | None:
    return find_file("data", "raw", "weather_features.csv")


def searched_market_paths() -> list[str]:
    return [str(r / "data" / "processed" / "market.parquet") for r in candidate_roots()[:8]]


def fill_short_gaps(df: pd.DataFrame, max_gap: int = MAX_GAP) -> pd.DataFrame:
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    for c in numeric:
        s = out[c]
        na = s.isna()
        if not na.any():
            continue
        run_id = na.ne(na.shift()).cumsum()
        run_len = na.groupby(run_id).transform("size")
        long_na = na & (run_len > max_gap)
        filled = s.interpolate(method="time", limit=max_gap)
        filled[long_na] = np.nan
        out[c] = filled
    return out


def build_market(energy_path: Path, weather_path: Path) -> pd.DataFrame:
    """Same cleaning contract as notebooks/01_eda.ipynb."""
    energy = pd.read_csv(energy_path)
    ts = pd.to_datetime(energy["time"], utc=True).dt.tz_convert(TZ)
    energy = energy.copy()
    energy.index = ts
    energy.index.name = "ts"
    energy = energy.drop(columns=["time"]).sort_index()
    present_dead = [c for c in DEAD_COLS if c in energy.columns]
    energy = energy.drop(columns=present_dead)
    energy = fill_short_gaps(energy)

    weather = pd.read_csv(weather_path)
    weather["city_name"] = weather["city_name"].str.strip()
    weather["ts"] = pd.to_datetime(weather["dt_iso"], utc=True).dt.tz_convert(TZ)
    weather.loc[(weather["pressure"] < 900) | (weather["pressure"] > 1100), "pressure"] = np.nan
    weather.loc[weather["wind_speed"] > 50, "wind_speed"] = np.nan
    num_w = ["temp", "humidity", "pressure", "wind_speed", "rain_1h", "clouds_all"]
    w_city = (
        weather.groupby(["ts", "city_name"], as_index=False)[num_w]
        .mean()
        .sort_values(["city_name", "ts"])
    )
    w_city["temp"] = w_city["temp"] - 273.15

    national = None
    for city, weight in CITY_WEIGHTS.items():
        piece = (
            w_city.loc[w_city["city_name"] == city, ["ts", *num_w]]
            .set_index("ts")
            .sort_index()
            * weight
        )
        national = piece if national is None else national.add(piece, fill_value=0)
    national = national.rename(
        columns={"temp": "temp_c", "clouds_all": "clouds"}
    )
    national = fill_short_gaps(national)

    market = energy.rename(columns=RENAME)
    keep = list(RENAME.values())
    market = market[keep].join(national, how="left")
    market["net_load"] = market["load_actual"] - market["gen_solar"] - market["gen_wind"]
    market["gen_hydro"] = market["gen_hydro_ror"] + market["gen_hydro_res"]
    return market


def build_and_save_market() -> Path:
    e, w = energy_csv(), weather_csv()
    if e is None or w is None:
        raise FileNotFoundError(
            "Need data/raw/energy_dataset.csv and weather_features.csv to rebuild."
        )
    market = build_market(e, w)
    # write next to the CSVs' repo, not cwd
    root = e.parents[1]  # .../data/raw/file → .../data → repo? raw is data/raw, parents[1] is data, parents[2] is repo
    root = e.resolve().parents[2]
    out = root / "data" / "processed" / "market.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(out)
    return out
