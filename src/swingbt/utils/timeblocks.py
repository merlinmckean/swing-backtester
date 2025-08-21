from __future__ import annotations
import pandas as pd
from typing import Iterator, Tuple, List




# src/swingbt/utils/timeblocks.py
def month_ends(dates: pd.Index) -> pd.DatetimeIndex:
    s = pd.Series(1, index=pd.DatetimeIndex(dates.unique()).sort_values())
    me = s.resample('ME').last().index   # was 'M'
    return me





def rolling_time_blocks(all_month_ends: pd.DatetimeIndex, train_years: int, test_months: int, embargo_months: int) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    n_train = train_years * 12
    for end_i in range(n_train, len(all_month_ends) - test_months + 1):
        tr_start = end_i - n_train
        tr_end = end_i # exclusive of test
        train = all_month_ends[tr_start:tr_end]
        test = all_month_ends[tr_end:tr_end + test_months]
        # Embargo: drop last `embargo_months` from train to avoid leakage
        if embargo_months > 0:
            train = train[:-embargo_months]
        yield train, test