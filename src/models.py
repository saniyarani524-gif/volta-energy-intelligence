"""Train / evaluate helpers for VOLTA forecasting.

All splits are temporal. Metrics are always MAE / RMSE / MAPE / R².
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.config import MODELS, QUANTILES, XGB_PARAMS, XGB_QUANTILE_PARAMS


def metrics(y_true, y_pred, name: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mape = float(np.mean(np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    out = {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    return out


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    tbl = pd.DataFrame(rows).set_index("model")
    return tbl[["MAE", "RMSE", "MAPE", "R2"]]


def fit_ridge(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def fit_xgb(X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> XGBRegressor:
    cfg = dict(XGB_PARAMS)
    if params:
        cfg.update(params)
    model = XGBRegressor(**cfg)
    model.fit(X, y, verbose=False)
    return model


def fit_quantiles(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: tuple[float, ...] = QUANTILES,
) -> dict[float, XGBRegressor]:
    models = {}
    for a in alphas:
        cfg = dict(XGB_QUANTILE_PARAMS)
        cfg.update({"objective": "reg:quantileerror", "quantile_alpha": a})
        m = XGBRegressor(**cfg)
        m.fit(X, y, verbose=False)
        models[a] = m
    return models


def walk_forward_monthly(
    frame: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    target: str,
    *,
    year: int = 2018,
    params: dict | None = None,
) -> pd.DataFrame:
    """Expanding-window backtest: at each month start, refit on all prior hours.

    Returns a DataFrame indexed like `frame` for `year`, with columns
    actual, pred, tso (price_da if present).
    """
    cfg = dict(XGB_PARAMS)
    if params:
        cfg.update(params)
    # slightly fewer trees — 12 refits
    cfg.setdefault("n_estimators", 300)

    pieces = []
    for month in range(1, 13):
        te_mask = (frame.index.year == year) & (frame.index.month == month)
        if not te_mask.any():
            continue
        cutoff = frame.index[te_mask][0]
        train = frame.loc[frame.index < cutoff]
        test = frame.loc[te_mask]
        if len(train) < 24 * 30:
            raise RuntimeError(f"not enough train history before {cutoff}")
        model = XGBRegressor(**cfg)
        model.fit(train[list(feature_cols)], train[target], verbose=False)
        pred = model.predict(test[list(feature_cols)])
        out = pd.DataFrame({"actual": test[target].values, "pred": pred}, index=test.index)
        if "price_da" in test.columns and target == "price_actual":
            out["tso"] = test["price_da"].values
        pieces.append(out)
    return pd.concat(pieces).sort_index()


def walk_forward_years(
    frame: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    target: str,
    years: tuple[int, ...] = (2016, 2017, 2018),
    params: dict | None = None,
) -> pd.DataFrame:
    """Expanding-window forecasts for several years (2015 is burn-in)."""
    parts = [
        walk_forward_monthly(frame, feature_cols, target, year=y, params=params)
        for y in years
    ]
    return pd.concat(parts).sort_index()


def save_xgb(model: XGBRegressor, name: str) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / name
    model.save_model(path)
    return path


def load_xgb(name: str) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(MODELS / name)
    return model
