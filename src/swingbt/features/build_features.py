from __future__ import annotations
import pandas as pd
import numpy as np
from .tech_indicators import rolling_vol, rsi


# Expect MultiIndex (date, symbol) with 'close', 'volume' (optional), and optional 'sector'




def _mom(df: pd.DataFrame, look: int, skip: int = 0) -> pd.Series:
    close_wide = df['close'].unstack()              # dates x symbols
    mom = close_wide.shift(skip).pct_change(look)   # compute per column
    return mom.stack()                               # back to (date, symbol)





def _drawdown(df: pd.DataFrame, win: int = 60) -> pd.Series:
    px = df['close']
    grp = px.groupby(level=1)
    roll_max = grp.transform(lambda s: s.rolling(win).max())
    dd = (px / roll_max) - 1.0
    return dd




def _dist_52w_high(df: pd.DataFrame) -> pd.Series:
    px = df['close']
    grp = px.groupby(level=1)
    high = grp.transform(lambda s: s.rolling(252).max())
    return (px / high) - 1.0




def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    feats = pd.DataFrame(index=df.index)


    # Momentum features
    feats['mom_12_1'] = _mom(df, look=252, skip=21)
    feats['mom_6_1'] = _mom(df, look=126, skip=21)
    feats['mom_3_1'] = _mom(df, look=63, skip=21)


    # Volatility
    close = df['close'].unstack()
    vol60 = close.pct_change().rolling(60).std() * np.sqrt(252)
    feats['vol_60d'] = vol60.stack()


    # Drawdown & 52w distance
    feats['dd_60d'] = _drawdown(df, 60)
    feats['dist_52w_high'] = _dist_52w_high(df)


    # RSI
    rsi14 = close.apply(rsi, axis=0, raw=False)
    feats['rsi_14'] = rsi14.stack()


    # Clean & drop rows with all‑nan features
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(how='all')


    return feats