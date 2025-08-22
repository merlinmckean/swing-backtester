from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import numpy as _np

def size_score_invvol(scores: _np.ndarray,
                      inv_vol: _np.ndarray,
                      max_w: float | None = 0.03) -> _np.ndarray:
    """
    Allocate weights proportional to (positive) score × inverse-volatility,
    with an optional per-name cap followed by renormalization.
    """
    raw = _np.clip(scores, 0, None) * inv_vol
    s = raw.sum()
    if s <= 0:
        return raw
    w = raw / s
    if max_w is not None:
        w = _np.minimum(w, max_w)
        w = w / (w.sum() + 1e-12)
    return w





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
    """
    Apply sector caps to a weight vector.

    Accepts `sectors` indexed either by symbol OR by (date, symbol).
    We align/derive a symbol->sector map and then cap sector totals to `cap`.
    """
    w = weights.copy()

    # --- Build a symbol->sector Series aligned to weights.index ---
    sym_index = w.index

    # Case A: sectors already indexed by symbol
    if sectors.index.nlevels == 1:
        sec_sym = sectors.reindex(sym_index)

    else:
        # Case B: sectors is MultiIndex (e.g., (date, symbol))
        # Reduce to a unique per-symbol mapping (first non-null sector)
        df_sec = sectors.reset_index()
        # Expect columns like ['date','symbol',0] or ['date','symbol','sector']
        # Find the sector column name
        sector_col = None
        for c in df_sec.columns:
            if c not in ("date", "symbol"):
                sector_col = c
                break
        if sector_col is None:
            # nothing we can do; return original weights
            return w

        # Deduplicate by symbol -> take first non-null sector
        df_sec = df_sec.dropna(subset=[sector_col])
        sec_map = df_sec.drop_duplicates(subset=["symbol"]).set_index("symbol")[sector_col]
        sec_sym = sym_index.to_series().map(sec_map)

    # If still all null, bail out gracefully
    if sec_sym.isna().all():
        return w

    # --- Apply cap per sector ---
    # Normalize to avoid drift from re-scaling
    total_before = w.sum()
    if total_before <= 0:
        return w

    for sec in sec_sym.dropna().unique():
        idx = sym_index[sec_sym.values == sec]
        total = w.loc[idx].sum()
        if total > cap:
            w.loc[idx] *= cap / max(total, 1e-12)

    s = w.sum()
    return w / s if s > 0 else w

def size_score_invvol(scores: _np.ndarray, inv_vol: _np.ndarray, max_w: float | None = 0.03) -> _np.ndarray:
    raw = _np.clip(scores, 0, None) * inv_vol
    s = raw.sum()
    if s <= 0:
        return raw
    w = raw / s
    if max_w is not None:
        w = _np.minimum(w, max_w)
        w = w / (w.sum() + 1e-12)
    return w

