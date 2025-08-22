# src/swingbt/models/xgb.py
from __future__ import annotations
from typing import Optional, Dict, Any
import xgboost as xgb
import numpy as np

def fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
    eval_metric: str | None = "rmse",
) -> xgb.XGBRegressor:
    """Fit an XGBRegressor with optional validation + early stopping.
    Works across XGBoost 1.x and 2.x (callbacks-first, fallback to legacy)."""
    params = (params or {}).copy()
    params.setdefault("n_estimators", 2000)
    params.setdefault("random_state", 42)

    # IMPORTANT: eval_metric must be on the constructor for XGB >= 2.x
    if eval_metric is not None:
        params["eval_metric"] = eval_metric  # explicit arg wins over params

    model = xgb.XGBRegressor(**params)

    fit_kwargs: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]

    # Prefer callback API (XGBoost ≥2.x)
    if early_stopping_rounds and "eval_set" in fit_kwargs:
        fit_kwargs["callbacks"] = [
            xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
                min_delta=0.0,
            )
        ]

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError as e:
        # Likely XGBoost 1.x which doesn’t accept 'callbacks'
        if "unexpected keyword argument 'callbacks'" in str(e):
            fit_kwargs.pop("callbacks", None)
            if early_stopping_rounds and "eval_set" in fit_kwargs:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            raise

    return model



# Append this to src/swingbt/models/xgb.py

from typing import Any
import numpy as np
import xgboost as xgb  # already imported above

def predict(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """
    Version-safe predict for XGBoost sklearn API across 1.x and 2.x,
    respecting early stopping if available.
    """
    # If we used callbacks with save_best=True (our fit_xgb does), the model
    # is already the best iteration; plain predict() is usually correct.
    # Still, handle explicit best-iteration paths for safety across versions.

    # Prefer iteration_range (XGBoost ≥2.x)
    best_iter: Any = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration_", None)

    if best_iter is not None:
        try:
            return model.predict(X, iteration_range=(0, int(best_iter) + 1))
        except TypeError:
            # Likely XGBoost 1.x where iteration_range isn't supported
            pass

    # XGBoost 1.x fallback: ntree_limit
    best_ntree = getattr(model, "best_ntree_limit", None)
    if best_ntree is not None:
        try:
            return model.predict(X, ntree_limit=int(best_ntree))
        except TypeError:
            pass

    # Final fallback
    return model.predict(X)
