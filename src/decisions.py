"""Explainable CHARGE / DISCHARGE / HOLD cards.

The LP in ``optimizer.py`` is optimal but mute. This module is the voice:

1. Solve the day-ahead LP on P50 prices.
2. Apply *day-level* risk scales (never hour-level — that would break SOC).
3. Stamp every hour with action, confidence, flags, expected €, and a why.

Flags do not silently rewrite physics except through the day-level scale.
A CHARGE hour that sits on a spike is *flagged*, not zeroed.

Confidence comes from the P10–P90 width relative to P50 (Phase 2 quantiles).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ACTION_MW_EPS,
    MIN_POSITION_SCALE,
    SPIKE_EUR,
    SPIKE_Z,
    UNCERTAINTY_HARD,
    UNCERTAINTY_SOFT,
)
from src.optimizer import DEFAULT_BATTERY, BatteryParams, solve_day


ACTIONS = ("CHARGE", "DISCHARGE", "HOLD")


def breakeven_spread(buy_price: float, battery: BatteryParams = DEFAULT_BATTERY) -> float:
    """Minimum (sell − buy) €/MWh that covers round-trip loss + degradation.

    Buy 1 MWh from the grid → store η → sell η² = round_trip MWh.
    Need  P_sell * rt >= P_buy + deg * rt
      ⇒  P_sell − P_buy >= P_buy * (1/rt − 1) + deg
    """
    rt = battery.round_trip
    return float(buy_price * (1.0 / rt - 1.0) + battery.degradation_eur_per_mwh)


def _rel_width(p10, p50, p90) -> np.ndarray:
    p10, p50, p90 = np.asarray(p10, float), np.asarray(p50, float), np.asarray(p90, float)
    width = np.clip(p90 - p10, 0.0, None)
    return width / np.clip(np.abs(p50), 1.0, None)


def _confidence(rel_w: np.ndarray) -> np.ndarray:
    # width 0 → 0.98; width 50% of price → ~0.15
    return np.clip(1.0 - rel_w / 0.50, 0.15, 0.98)


def _day_scale(rel_w: np.ndarray, spread: float, be: float) -> tuple[float, list[str]]:
    """Single scale for the whole day so SOC physics stay feasible."""
    flags: list[str] = []
    if not np.isfinite(spread) or spread < be:
        flags.append("BELOW_BREAKEVEN")
        return 0.0, flags
    rw = float(np.nanmedian(rel_w)) if len(rel_w) else 0.0
    if rw >= UNCERTAINTY_HARD:
        flags.append("HIGH_UNCERTAINTY")
        return MIN_POSITION_SCALE, flags
    if rw >= UNCERTAINTY_SOFT:
        flags.append("HIGH_UNCERTAINTY")
        return 0.5, flags
    return 1.0, flags


def _why(
    action: str,
    ts: pd.Timestamp,
    mw: float,
    price: float,
    day_lo: float,
    day_hi: float,
    spread: float,
    be: float,
    conf: float,
    flags: list[str],
) -> str:
    hh = ts.strftime("%H:%M")
    if action == "HOLD" and mw < ACTION_MW_EPS:
        if "BELOW_BREAKEVEN" in flags:
            return (
                f"HOLD at {hh}. Forecast spread €{spread:.1f}/MWh does not clear "
                f"round-trip + degradation (need €{be:.1f}). Sitting this hour out."
            )
        if "HIGH_UNCERTAINTY" in flags:
            return (
                f"HOLD at {hh}. Quantile band is wide relative to P50 "
                f"(confidence {conf:.0%}) — not a night to size up."
            )
        return (
            f"HOLD at {hh}. LP has no flow here (forecast €{price:.1f}, "
            f"day range €{day_lo:.1f}–€{day_hi:.1f})."
        )
    if action == "CHARGE":
        extra = " SPIKE flag: charging into a rich hour — review." if "SPIKE" in flags else ""
        return (
            f"CHARGE {mw:.0f} MW at {hh}. P50 €{price:.1f}/MWh is near the day's floor "
            f"(€{day_lo:.1f}–€{day_hi:.1f}). Fill now; sell the peak. "
            f"Spread €{spread:.1f} vs breakeven €{be:.1f}. Confidence {conf:.0%}.{extra}"
        )
    extra = " Spike hour — this is the print we wanted." if "SPIKE" in flags else ""
    return (
        f"DISCHARGE {mw:.0f} MW at {hh} into P50 €{price:.1f}/MWh "
        f"(day €{day_lo:.1f}–€{day_hi:.1f}). Spread €{spread:.1f} vs breakeven €{be:.1f}. "
        f"Confidence {conf:.0%}.{extra}"
    )


@dataclass
class DayDecision:
    cards: pd.DataFrame
    scale: float
    day_flags: list[str]
    expected_eur: float
    spread_eur: float
    breakeven_eur: float


def decide_day(
    p50: pd.Series,
    p10: pd.Series | None = None,
    p90: pd.Series | None = None,
    battery: BatteryParams = DEFAULT_BATTERY,
) -> DayDecision:
    """One Madrid-local day → ranked action cards.

    ``p50`` is the price the LP sees. Quantiles are optional; without them
    confidence is 0.50 and uncertainty flags stay off.
    """
    p50 = p50.astype(float)
    if p50.isna().any():
        raise ValueError("p50 contains NaN")
    idx = p50.index
    T = len(p50)
    if p10 is None:
        p10 = p50
        p90 = p50
        have_q = False
    else:
        p10 = p10.reindex(idx).astype(float)
        p90 = p90.reindex(idx).astype(float)
        have_q = True
        if p10.isna().any() or p90.isna().any():
            p10 = p10.fillna(p50)
            p90 = p90.fillna(p50)

    planned = solve_day(p50.values, battery)
    rel_w = _rel_width(p10.values, p50.values, p90.values) if have_q else np.zeros(T)
    conf = _confidence(rel_w) if have_q else np.full(T, 0.50)

    charge, discharge = planned.charge_mw.copy(), planned.discharge_mw.copy()
    # VWAP on the *forecast* the desk actually traded
    dis_sum, ch_sum = float(discharge.sum()), float(charge.sum())
    vwap_sell = float(np.dot(p50.values, discharge) / dis_sum) if dis_sum > 1e-6 else np.nan
    vwap_buy = float(np.dot(p50.values, charge) / ch_sum) if ch_sum > 1e-6 else np.nan
    spread = (vwap_sell - vwap_buy) if np.isfinite(vwap_sell) and np.isfinite(vwap_buy) else 0.0
    be = breakeven_spread(vwap_buy if np.isfinite(vwap_buy) else float(p50.min()), battery)

    scale, day_flags = _day_scale(rel_w, spread, be)
    charge *= scale
    discharge *= scale
    soc = planned.soc_mwh.copy()
    if scale == 0.0:
        mid = 0.5 * (battery.e_min + battery.e_max)
        soc = np.full_like(soc, mid)
    elif scale != 1.0:
        # linear scale around the cyclic SOC path: E' = E_start + scale*(E - E_start)
        soc = soc[0] + scale * (soc - soc[0])

    day_lo, day_hi = float(p50.min()), float(p50.max())
    mu, sd = float(p50.mean()), float(p50.std(ddof=0) or 1.0)
    spike = (p50.values >= SPIKE_EUR) | (p50.values >= mu + SPIKE_Z * sd)

    rows = []
    for i, ts in enumerate(idx):
        ch, dis = float(charge[i]), float(discharge[i])
        if dis >= ACTION_MW_EPS and dis >= ch:
            action, mw = "DISCHARGE", dis
        elif ch >= ACTION_MW_EPS:
            action, mw = "CHARGE", ch
        else:
            action, mw = "HOLD", 0.0

        flags = list(day_flags)
        if spike[i]:
            flags.append("SPIKE")
        # unique, stable order
        seen, ordered = set(), []
        for f in flags:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
        flags = ordered

        if action == "DISCHARGE":
            expected = mw * float(p50.iloc[i]) - battery.degradation_eur_per_mwh * mw
        elif action == "CHARGE":
            expected = -mw * float(p50.iloc[i])
        else:
            expected = 0.0

        rows.append(
            {
                "ts": ts,
                "action": action,
                "mw": mw,
                "soc_mwh": float(soc[i]),
                "price_p50": float(p50.iloc[i]),
                "price_p10": float(p10.iloc[i]),
                "price_p90": float(p90.iloc[i]),
                "band_width": float(p90.iloc[i] - p10.iloc[i]),
                "confidence": float(conf[i]),
                "expected_eur": float(expected),
                "flags": "|".join(flags),
                "why": _why(
                    action, ts, mw, float(p50.iloc[i]), day_lo, day_hi,
                    spread, be, float(conf[i]), flags,
                ),
            }
        )

    cards = pd.DataFrame(rows).set_index("ts").sort_index()
    # rank: money first, HOLDs last
    score = cards["expected_eur"].abs()
    score = score.where(cards["action"] != "HOLD", -1.0)
    cards["rank"] = score.rank(ascending=False, method="first").astype(int)
    return DayDecision(
        cards=cards,
        scale=scale,
        day_flags=day_flags,
        expected_eur=float(cards["expected_eur"].sum()),
        spread_eur=float(spread),
        breakeven_eur=float(be),
    )


def decide_range(
    p50: pd.Series,
    p10: pd.Series | None = None,
    p90: pd.Series | None = None,
    battery: BatteryParams = DEFAULT_BATTERY,
) -> pd.DataFrame:
    """Concatenate hourly cards for every calendar day in the series."""
    p50 = p50.astype(float)
    pieces = []
    for day, s in p50.groupby(p50.index.date):
        q10 = p10.reindex(s.index) if p10 is not None else None
        q90 = p90.reindex(s.index) if p90 is not None else None
        d = decide_day(s, q10, q90, battery)
        extra = d.cards.copy()
        extra["day"] = pd.Timestamp(day)
        extra["scale"] = d.scale
        extra["day_expected_eur"] = d.expected_eur
        extra["day_spread_eur"] = d.spread_eur
        pieces.append(extra)
    return pd.concat(pieces).sort_index()
