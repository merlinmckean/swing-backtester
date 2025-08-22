# src/swingbt/backtest/evaluation.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def spearman_ic(pred, actual) -> float:
    """
    Cross-sectional Spearman rank correlation (Information Coefficient).
    Accepts pandas Series (preferred) or numpy arrays. If Series, aligns by index.
    """
    if isinstance(pred, pd.Series) and isinstance(actual, pd.Series):
        a, b = actual.align(pred, join="inner")
        mask = a.notna() & b.notna()
        if mask.sum() < 3:
            return np.nan
        return float(spearmanr(a[mask].values, b[mask].values).correlation)

    pred = np.asarray(pred)
    actual = np.asarray(actual)
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(actual[mask], pred[mask]).correlation)

def equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).fillna(0.0)
    return (1.0 + r).cumprod()

def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())  # negative number (e.g., -0.12)

def perf_summary(returns: pd.Series, freq: int) -> dict:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    mu = r.mean() * freq
    sd = r.std(ddof=1) * np.sqrt(freq)
    sharpe = mu / (sd + 1e-12)
    mdd = max_drawdown(r)
    return dict(ann_return=float(mu), ann_vol=float(sd), sharpe=float(sharpe), max_dd=float(mdd))
