# src/swingbt/models/cv.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from ..backtest.evaluation import spearman_ic
from .xgb import fit_xgb, predict

def time_folds(dates: pd.Index, n_folds: int = 3, embargo: int = 1) -> List[Tuple[pd.Index, pd.Index]]:
    d = pd.DatetimeIndex(sorted(pd.Index(dates).unique()))
    n = len(d)
    folds, sz = [], max(1, n // n_folds)
    for k in range(n_folds):
        te_start = k * sz
        te_end = n if k == n_folds - 1 else (k + 1) * sz
        test = d[te_start:te_end]
        train = d[:te_start]
        if embargo > 0 and len(train) > embargo:
            train = train[:-embargo]
        if len(train) < 3 or len(test) < 1:
            continue
        folds.append((train, test))
    return folds

def _split_train_val_by_date(X: pd.DataFrame, y: pd.Series, val_last_k: int = 2):
    dates = pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
    if len(dates) <= val_last_k:
        return X, y, None, None
    val_dates = dates[-val_last_k:]
    tr = X.index.get_level_values(0).isin(dates[:-val_last_k])
    va = X.index.get_level_values(0).isin(val_dates)
    X_tr, y_tr = X.loc[tr], y.loc[tr]
    X_val, y_val = X.loc[va], y.loc[va]
    com_tr = X_tr.index.intersection(y_tr.index)
    com_va = X_val.index.intersection(y_val.index)
    return X_tr.loc[com_tr], y_tr.loc[com_tr], X_val.loc[com_va], y_val.loc[com_va]

def cv_ic_grid(X: pd.DataFrame, y: pd.Series, grid: List[dict]) -> dict:
    dates = X.index.get_level_values(0)
    folds = time_folds(dates, n_folds=3, embargo=1)
    best, best_ic = None, -1e9
    for params in grid:
        ics = []
        for tr_me, te_me in folds:
            X_tr0 = X.loc[X.index.get_level_values(0).isin(tr_me)]
            y_tr0 = y.loc[y.index.get_level_values(0).isin(tr_me)]
            com_tr0 = X_tr0.index.intersection(y_tr0.index)
            if len(com_tr0) < 20:
                continue
            X_tr0, y_tr0 = X_tr0.loc[com_tr0], y_tr0.loc[com_tr0]

            # inner train/val split for early stopping
            Xt, yt, Xv, yv = _split_train_val_by_date(X_tr0, y_tr0, val_last_k=2)

            m = fit_xgb(Xt, yt, params=params, X_val=Xv, y_val=yv, early_stopping_rounds=30)

            X_te = X.loc[X.index.get_level_values(0).isin(te_me)]
            y_te = y.loc[y.index.get_level_values(0).isin(te_me)]
            com_te = X_te.index.intersection(y_te.index)
            if len(com_te) < 5:
                continue
            X_te, y_te = X_te.loc[com_te], y_te.loc[com_te]

            for t in pd.DatetimeIndex(sorted(y_te.index.get_level_values(0).unique())):
                Xt = X_te.loc[X_te.index.get_level_values(0) == t]
                yt = y_te.loc[y_te.index.get_level_values(0) == t]
                if Xt.empty or yt.empty:
                    continue
                preds = predict(m, Xt)
                ic = spearman_ic(pd.Series(preds, index=Xt.index.get_level_values(1)), yt.droplevel(0))
                if np.isfinite(ic):
                    ics.append(ic)
        mean_ic = float(np.nanmean(ics)) if ics else -1e9
        if mean_ic > best_ic:
            best_ic, best = mean_ic, params
    return best or {}

