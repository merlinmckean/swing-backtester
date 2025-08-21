from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional


# Expected schema for prices: MultiIndex (date, symbol) with columns: close, volume, open, high, low, adj_close (optional), sector (optional), mcap (optional)




def load_prices(path: str | Path, source: str = "parquet") -> pd.DataFrame:
    path = Path(path)
    if source == "parquet":
        df = pd.read_parquet(path)
    elif source == "csv":
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raise ValueError(f"Unsupported source: {source}")


    # Standardize index: (date, symbol)
    if not isinstance(df.index, pd.MultiIndex):
        if {"date", "symbol"}.issubset(df.columns):
            df = df.set_index(["date", "symbol"]).sort_index()
        else:
            raise ValueError("Data must have MultiIndex (date, symbol) or columns ['date','symbol']")


    # Ensure float dtype where relevant
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


    return df




def add_month_end_flag(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dates = idx.get_level_values(0)
    is_me = dates.to_series().groupby(level=0).transform(lambda x: (x.index == x.index.to_series().resample('M').max().values).astype(int))
    # The above is messy; simpler: compute month-end per date
    d = dates.to_series().dt.to_period('M').dt.to_timestamp('M')
    df = df.copy()
    df["is_month_end"] = (dates == d.values).astype(int).values
    return df