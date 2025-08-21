from __future__ import annotations
import pandas as pd




def make_excess_return_targets(prices: pd.DataFrame, benchmark: str, lookahead_days: int) -> pd.Series:
    """Compute horizon-aligned excess returns R_i - R_bench over next lookahead_days.
    prices: MultiIndex (date, symbol) with 'close'
    benchmark: symbol name present in index level 1
    """
    df = prices['close'].unstack()
    fwd = df.pct_change(lookahead_days).shift(-lookahead_days)
    y = (fwd.sub(fwd[benchmark], axis=0)).stack()
    y.name = 'y'
    return y