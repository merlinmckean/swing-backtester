# src/swingbt/models/cv.py
# NOTE: keep __future__ import first or just omit it entirely if Py>=3.9
# from __future__ import annotations

from typing import List, Tuple, Dict, Any, Iterable
import numpy as np
import pandas as pd

from ..backtest.evaluation import spearman_ic
from .xgb import fit_xgb, predict


def time_folds(dates: pd.Index, n_folds: int = 3, embargo: int = 1) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Build time-ordered train/test folds over unique dates with an optional embargo.
    Returns a list of (train_dates, test_dates).
    """
    d = pd.DatetimeIndex(sorted(pd.Index(dates).unique()))
    n = len(d)
    folds: List[Tuple[pd.Index, pd.Index]] = []
    sz = max(1, n // n_folds)
    for k in range(n_folds):
        te_start = k * sz
        te_end = n if k == n_folds - 1 else (k + 1) * sz
        test = d[te_start:te_end]
        # train is everything before/after with an embargo gap
        tr_left_end = max(0, te_start - embargo)
        tr_right_start = min(n, te_end + embargo)
        train = d[:tr_left_end].append(d[tr_right_start:])
        folds.append((train, test))
    return folds


def cv_ic_grid(
    X: pd.DataFrame,
    y: pd.Series,
    grid: Iterable[Dict[str, Any]],
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Cross-validate a grid of XGB params using cross-sectional Spearman IC.
    X: MultiIndex (date, symbol) DataFrame (features)
    y: MultiIndex (date, symbol) Series (targets)
    Returns the best param dict (and the caller fits the final model).
    """
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("X index must be MultiIndex (date, symbol)")
    if not isinstance(y.index, pd.MultiIndex):
        raise ValueError("y index must be MultiIndex (date, symbol)")

    # Build folds over unique dates in y (safer if X has a superset)
    dates = pd.DatetimeIndex(sorted(y.index.get_level_values(0).unique()))
    folds = time_folds(dates, n_folds=n_folds, embargo=1)

    best_params: Dict[str, Any] = {}
    best_ic = -1e9

    for raw in grid:
        # Enforce fast, safe defaults
        params = dict(raw)
        params.setdefault("n_estimators", 500)
        params.setdefault("n_jobs", -1)
        params.setdefault("tree_method", "hist")
        params.setdefault("max_bin", 256)

        fold_ics: List[float] = []

        for i, (tr_dates, te_dates) in enumerate(folds, start=1):
            # slice by dates
            X_tr = X.loc[X.index.get_level_values(0).isin(tr_dates)]
            y_tr = y.loc[y.index.get_level_values(0).isin(tr_dates)]
            X_te = X.loc[X.index.get_level_values(0).isin(te_dates)]
            y_te = y.loc[y.index.get_level_values(0).isin(te_dates)]

            if X_tr.empty or y_tr.empty or X_te.empty or y_te.empty:
                fold_ics.append(np.nan)
                continue

            # Fit with lower patience for speed
            m = fit_xgb(
                X_tr.values, y_tr.values,
                params=params,
                X_val=X_te.values, y_val=y_te.values,
                early_stopping_rounds=10,
                eval_metric="rmse",
            )

            # Evaluate IC per test date, then average
            ics: List[float] = []
            for t in pd.DatetimeIndex(sorted(y_te.index.get_level_values(0).unique())):
                Xt = X_te.loc[X_te.index.get_level_values(0) == t]
                yt = y_te.loc[y_te.index.get_level_values(0) == t]
                if Xt.empty or yt.empty:
                    continue
                preds = predict(m, Xt.values)
                preds_s = pd.Series(preds, index=Xt.index.get_level_values(1))
                ic = spearman_ic(preds_s, yt.droplevel(0))
                if np.isfinite(ic):
                    ics.append(float(ic))

            fold_ic = float(np.nanmean(ics)) if len(ics) else np.nan
            fold_ics.append(fold_ic)

            # Simple prune: if the first two folds are both non-positive, bail early
            if i == 2 and all(np.nan_to_num(v, nan=-1e9) <= 0.0 for v in fold_ics[:2]):
                break

        mean_ic = float(np.nanmean([v for v in fold_ics if np.isfinite(v)])) if any(np.isfinite(v) for v in fold_ics) else -1e9
        if mean_ic > best_ic:
            best_ic, best_params = mean_ic, params

    return best_params or {}
