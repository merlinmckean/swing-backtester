from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata




def inverse_vol_weights(vol: np.ndarray, sel: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.clip(vol, 1e-6, np.inf)
    w = iv * sel
    s = w.sum()
    return w / s if s > 0 else w




def top_quantile_mask(scores: np.ndarray, q: float) -> np.ndarray:
    ranks = rankdata(scores) / len(scores)
    thresh = np.quantile(ranks, 1 - q)
    return (ranks >= thresh).astype(int)




def sector_cap_weights(weights: pd.Series, sectors: pd.Series, cap: float = 0.25) -> pd.Series:
    w = weights.copy()
    for sec, idx in sectors.groupby(sectors).groups.items():
        total = w.loc[idx].sum()
        if total > cap:
            w.loc[idx] *= cap / total
    return w / w.sum() if w.sum() > 0 else w