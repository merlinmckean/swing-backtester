# src/swingbt/models/xgb.py
from __future__ import annotations
from typing import Optional, Dict, Any, Union
import numpy as np
import xgboost as xgb

XGBModelT = Union[xgb.XGBRegressor, xgb.Booster]

def fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
    eval_metric: str | None = "rmse",
) -> XGBModelT:
    """
    Fit an XGBoost model with optional validation + early stopping.

    Tries in order:
      1) sklearn wrapper + callbacks (XGBoost >= 2.x)
      2) sklearn wrapper + early_stopping_rounds kwarg (XGBoost 1.x)
      3) sklearn wrapper plain fit (no ES)
      4) Native xgboost.train fallback (always supports ES)
    Returns either an XGBRegressor or a Booster; use `predict()` below for inference.
    """
    p = dict(params or {})
    p.setdefault("n_estimators", 500)
    p.setdefault("random_state", 42)

    # In >=2.x, eval_metric must be on the constructor for sklearn API.
    if eval_metric is not None:
        p.setdefault("eval_metric", eval_metric)

    # ---------- Attempts 1–3: sklearn wrapper ----------
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
    try:
        model = xgb.XGBRegressor(**p)
    except TypeError:
        model = None  # bizarre env; skip to native

    last_err: Exception | None = None
    if model is not None:
        # (1) callbacks path
        if early_stopping_rounds and eval_set is not None:
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[xgb.callback.EarlyStopping(
                        rounds=early_stopping_rounds,
                        save_best=True,
                        min_delta=0.0,
                    )],
                )
                return model
            except TypeError as e:
                last_err = e  # continue

        # (2) legacy kwarg path
        if early_stopping_rounds and eval_set is not None:
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                )
                return model
            except TypeError as e:
                last_err = e  # continue

        # (3) plain fit
        try:
            if eval_set is not None:
                model.fit(X_train, y_train, eval_set=eval_set)
            else:
                model.fit(X_train, y_train)
            return model
        except Exception as e:
            last_err = e  # fall through to native API

    # ---------- Attempt 4: native xgb.train fallback ----------
    # Map sklearn-style params to native params
    num_boost_round = int(p.pop("n_estimators", 2000))
    rng = p.pop("random_state", 42)

    native_params = {
        # objective default (regression); if you're ranking/classifying, grid can override
        "objective": p.pop("objective", "reg:squarederror"),
        "seed": rng,
        # carry through common params when present:
        **{k: v for k, v in p.items()
           if k in {
               "max_depth", "learning_rate", "subsample", "colsample_bytree",
               "colsample_bylevel", "colsample_bynode", "gamma", "min_child_weight",
               "reg_alpha", "reg_lambda", "tree_method", "max_bin", "grow_policy",
               "eval_metric"
           }},
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = []
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, "train"), (dval, "eval")]

    booster = xgb.train(
        native_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals if evals else None,
        early_stopping_rounds=early_stopping_rounds if (early_stopping_rounds and evals) else None,
        verbose_eval=False,
    )
    return booster


def predict(model: XGBModelT, X: np.ndarray) -> np.ndarray:
    """
    Version-safe predict for both XGBRegressor (sklearn API) and Booster (native API),
    respecting early stopping when available.
    """
    # Native Booster
    if isinstance(model, xgb.Booster):
        d = xgb.DMatrix(X)
        # Prefer best_iteration when available; else best_ntree_limit; else full
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            return model.predict(d, iteration_range=(0, int(best_iter) + 1))
        best_ntree = getattr(model, "best_ntree_limit", None)
        if best_ntree is not None:
            return model.predict(d, ntree_limit=int(best_ntree))
        return model.predict(d)

    # sklearn wrapper
    best_iter = getattr(model, "best_iteration", None) or getattr(model, "best_iteration_", None)
    if best_iter is not None:
        try:
            return model.predict(X, iteration_range=(0, int(best_iter) + 1))
        except TypeError:
            pass
    best_ntree = getattr(model, "best_ntree_limit", None)
    if best_ntree is not None:
        try:
            return model.predict(X, ntree_limit=int(best_ntree))
        except TypeError:
            pass
    return model.predict(X)
