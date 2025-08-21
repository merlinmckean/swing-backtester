from __future__ import annotations
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from ..utils.portfolio import inverse_vol_weights, top_quantile_mask, sector_cap_weights
from ..utils.costs import apply_trading_costs
from ..utils.timeblocks import month_ends, rolling_time_blocks
from ..features.build_features import build_features
from ..targets.make_targets import make_excess_return_targets
from ..models.xgb import fit_xgb, predict
from .evaluation import spearman_ic, perf_summary


# Periods-per-year map for annualizing stats
FREQ_MAP = {1: 252, 5: 52, 21: 12}


def run_backtest(
    prices: pd.DataFrame,
    benchmark: str,
    lookahead_days: int,
    rebalance_days: int,
    cost_bps: float,
    train_years: int,
    test_months: int,
    embargo_months: int,
    top_q: float = 0.30,
    sector_caps: bool = True,
) -> Dict:
    """
    Walk-forward backtest on (date,symbol)-indexed prices.
    Builds features/labels once, trains XGB on rolling train windows,
    scores cross-sections at test month-ends, constructs a top-quantile
    inverse-vol portfolio, and aggregates returns with simple costs.
    """
    # --- Build features/targets (point-in-time) ---
    X = build_features(prices)  # (date,symbol) -> feature columns
    y = make_excess_return_targets(prices, benchmark=benchmark, lookahead_days=lookahead_days)  # Series 'y'

    # Align to shared index
    idx = X.index.intersection(y.index)
    if len(idx) == 0:
        return {
            "ic_mean": float("nan"),
            "ic_ir": float("nan"),
            "stats": {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0},
            "returns_net": pd.Series(dtype=float),
        }

    X = X.loc[idx]
    y = y.loc[idx]

    # Month-ends for walk-forward splits
    all_me = month_ends(pd.DatetimeIndex(sorted(set(idx.get_level_values(0)))))

    ics: List[float] = []
    ret_list: List[pd.Series] = []
    turnover_list: List[pd.Series] = []

    # --- Rolling train/test over month-ends ---
    for train_me, test_me in rolling_time_blocks(all_me, train_years, test_months, embargo_months):
        train_dates = set(train_me)
        test_dates = set(test_me)

        tr_idx = [i for i in X.index if i[0].normalize() in train_dates]
        te_idx = [i for i in X.index if i[0].normalize() in test_dates]
        if not tr_idx or not te_idx:
            continue

        X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
        model = fit_xgb(X_tr, y_tr)

        # Score each test month cross-section
        for t in sorted(set(pd.DatetimeIndex([i[0] for i in te_idx]))):
            rows_t = [i for i in te_idx if i[0] == t]
            if not rows_t:
                continue

            X_t = X.loc[rows_t]
            y_t = y.loc[rows_t]

            preds = predict(model, X_t)
            ics.append(spearman_ic(preds, y_t.values))

            # Select & size
            sel = top_quantile_mask(preds, q=top_q)
            vol = X_t["vol_60d"].values if "vol_60d" in X_t.columns else np.ones(len(X_t))
            w = inverse_vol_weights(vol, sel)
            w_ser = pd.Series(w, index=X_t.index.get_level_values(1))

            # Optional sector caps if sector column exists
            if sector_caps and "sector" in prices.columns:
                sectors = prices.loc[rows_t, "sector"]
                w_ser = sector_cap_weights(w_ser, sectors)

            # Realized next-period excess return (already excess vs benchmark)
            ret = float(np.dot(w_ser.values, y_t.values))
            ret_list.append(pd.Series({t: ret}))

            # Turnover proxy (all-in/out per rebalance → ~sum|w|)
            turnover_list.append(pd.Series({t: float(w_ser.abs().sum())}))

    # --- Aggregate & apply costs ---
    if ret_list:
        returns = pd.concat(ret_list).sort_index()
    else:
        returns = pd.Series(dtype=float)

    if turnover_list:
        turnover = (
            pd.concat(turnover_list)
            .sort_index()
            .reindex_like(returns)
            .fillna(method="ffill")
            .fillna(0.0)
        )
    else:
        turnover = pd.Series(0.0, index=returns.index)

    returns_net = pd.Series(
        apply_trading_costs(returns.values, turnover.values, cost_bps),
        index=returns.index,
    )

    # --- Stats ---
    freq = FREQ_MAP.get(rebalance_days, 12)
    stats = perf_summary(returns_net if len(returns_net) else pd.Series([0.0]), freq=freq)

    # --- Result dict ---
    ic_mean = float(np.mean(ics)) if ics else float("nan")
    ic_ir = float(np.mean(ics) / (np.std(ics) + 1e-12)) if ics else float("nan")

    return {
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "stats": stats,
        "returns_net": returns_net,
    }
