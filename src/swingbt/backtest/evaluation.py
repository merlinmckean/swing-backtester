from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata




def spearman_ic(pred: np.ndarray, y: np.ndarray) -> float:
    x = rankdata(pred)
    y_r = rankdata(y)
    c = np.corrcoef(x, y_r)[0, 1]
    return float(c)




def perf_summary(returns: pd.Series, freq: int) -> dict:
    mu = returns.mean() * freq
    sd = returns.std() * np.sqrt(freq)
    sharpe = mu / (sd + 1e-12)
    dd = (returns.add(1).cumprod().cummax() / returns.add(1).cumprod() - 1).min()
    return dict(ann_return=mu, ann_vol=sd, sharpe=sharpe, max_dd=dd)