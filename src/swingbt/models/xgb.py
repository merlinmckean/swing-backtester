from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor




DEFAULT_PARAMS = dict(
n_estimators=800,
max_depth=4,
learning_rate=0.03,
subsample=0.6,
colsample_bytree=0.6,
min_child_weight=20,
reg_alpha=1.0,
reg_lambda=2.0,
objective="reg:squarederror",
n_jobs=-1,
)




def fit_xgb(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None, params: dict | None = None) -> XGBRegressor:
    params = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBRegressor(**params)
    model.fit(X.values, y.values, sample_weight=sample_weight)
    return model




def predict(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X.values)