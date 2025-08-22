from __future__ import annotations
import pandas as pd


def make_excess_return_targets(prices: pd.DataFrame, benchmark: str, lookahead_days: int) -> pd.Series:
    df = prices['close'].unstack()
    fwd = df.pct_change(lookahead_days).shift(-lookahead_days)
    y = (fwd.sub(fwd[benchmark], axis=0)).stack()
    y.name = 'y'
    # winsorize at ~5σ
    m, s = y.mean(), y.std(ddof=1)
    lo, hi = m - 5*s, m + 5*s
    return y.clip(lo, hi)
