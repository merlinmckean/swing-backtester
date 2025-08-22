# src/swingbt/backtest/run_backtest.py
from __future__ import annotations
from typing import Dict, List
import numpy as _np
import pandas as _pd

from ..features.build_features import build_features
from ..targets.make_targets import make_excess_return_targets
from ..models.xgb import fit_xgb, predict
from ..models.cv import cv_ic_grid
from ..utils.timeblocks import month_ends, rolling_time_blocks
from ..utils.portfolio import (
    inverse_vol_weights,
    top_quantile_mask,
    sector_cap_weights,
    size_score_invvol,   # <-- import the new sizing helper
)
from ..utils.costs import apply_trading_costs
from .evaluation import spearman_ic, perf_summary

# Annualization map by rebalance days
_FREQ_MAP = {1: 252, 5: 52, 21: 12}


def _freq_from_days(rebalance_days: int) -> int:
    return _FREQ_MAP.get(int(rebalance_days), 12)


def run_backtest(prices: _pd.DataFrame, horizon: str, cfg: Dict) -> Dict:
    """
    prices: MultiIndex (date, symbol) DataFrame with at least 'close', optional 'sector'
    horizon: key in cfg['horizons'] ('daily' | 'weekly' | 'monthly')
    cfg: loaded params.yaml as dict
    """
    # ---- Horizon params ----
    hz = cfg["horizons"][horizon]
    lookahead_days = int(hz["lookahead_days"])
    rebalance_days = int(hz["rebalance_days"])
    cost_bps = float(hz.get("cost_bps_per_trade", 5.0))
    benchmark = cfg.get("universe", {}).get("benchmark", "SPY")

    # ---- Validation params ----
    val = cfg["validation"]
    train_years = int(val["train_years"])
    test_months = int(val["test_months"])
    embargo_months = int(val["embargo_months"])

    # ---- Portfolio params ----
    port = cfg["portfolio"]
    top_q = float(port["top_quantile"])
    use_sector_caps = bool(port.get("sector_caps", False))
    max_w_name = float(port.get("max_weight_per_name", 0.03))

    # ---- Features & labels ----
    X = build_features(prices)  # index: (date, symbol)
    y = make_excess_return_targets(prices, benchmark, lookahead_days)  # index: (date, symbol)

    # Month-ends from feature dates
    all_dates = _pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
    all_me = month_ends(all_dates)

    ics: List[float] = []
    ret_list: List[_pd.Series] = []
    turn_list: List[_pd.Series] = []
    last_w: _pd.Series | None = None

    # ---- Walk-forward ----
    for train_me, test_me in rolling_time_blocks(all_me, train_years, test_months, embargo_months):
        # TRAIN: filter each side independently, then intersect exact rows
        X_tr = X.loc[X.index.get_level_values(0).isin(train_me)]
        y_tr = y.loc[y.index.get_level_values(0).isin(train_me)]
        if X_tr.empty or y_tr.empty:
            continue
        common_tr = X_tr.index.intersection(y_tr.index)
        if len(common_tr) < 10:
            continue
        X_tr = X_tr.loc[common_tr]
        y_tr = y_tr.loc[common_tr]

        grid = [
            {"max_depth": d, "learning_rate": lr, "min_child_weight": mcw, "subsample": 0.8, "colsample_bytree": cs, "n_estimators": 500}
            for d in (3, 4)
            for lr in (0.01, 0.03)
            for mcw in (10, 20)
            for cs in (0.6, 0.8)
        ]
        best = cv_ic_grid(X_tr, y_tr, grid)
        model = fit_xgb(X_tr, y_tr, params=best)

        # TEST: score each test month-end t
        for t in test_me:
            X_t = X.loc[X.index.get_level_values(0) == t]
            y_t = y.loc[y.index.get_level_values(0) == t]
            if X_t.empty or y_t.empty:
                continue

            common_t = X_t.index.intersection(y_t.index)
            if len(common_t) < 5:
                continue
            X_t = X_t.loc[common_t]
            y_t = y_t.loc[common_t]  # (date=t, symbol)

            # ---- Drop benchmark from candidate set (labels are excess vs SPY) ----
            sym_index = X_t.index.get_level_values(1)
            if benchmark in sym_index:
                mask_not_bench = sym_index != benchmark
                # filter rows to non-benchmark symbols
                keep_syms = sym_index[mask_not_bench]
                X_t = X_t.loc[X_t.index.get_level_values(1).isin(keep_syms)]
                y_t = y_t.loc[y_t.index.get_level_values(1).isin(keep_syms)]
                sym_index = keep_syms

            # ---- Predict scores aligned by symbol ----
            preds = predict(model, X_t)
            preds_s = _pd.Series(preds, index=sym_index)

            # IC on cross-section (align to symbols)
            ics.append(spearman_ic(preds_s, y_t.droplevel(0)))

            # ---- Sizing inputs (this is where we inject score × inv-vol sizing) ----
            if "vol_60d" in X_t.columns:
                vol = X_t["vol_60d"].values
            else:
                # conservative fallback if vol_60d missing
                vol = (
                    X_t.groupby(level=1)["close"].std()
                      .reindex(sym_index).fillna(1.0).values
                )

            # top-quantile selection by score
            sel = top_quantile_mask(preds_s.values, top_q).astype(bool)
            if sel.sum() == 0:
                continue

            scores_sel = preds_s.values[sel]
            inv_vol_sel = 1.0 / _np.clip(vol[sel], 1e-6, _np.inf)

            # >>> injected block: score × inverse-vol with per-name cap <<<
            w_sel = size_score_invvol(scores_sel, inv_vol_sel, max_w=max_w_name)
            w_ser = _pd.Series(w_sel, index=sym_index[sel])

            # ---- Optional sector caps ----
            if use_sector_caps and "sector" in prices.columns:
                try:
                    sec_t = prices["sector"].xs(t).reindex(w_ser.index)
                    w_ser = sector_cap_weights(w_ser, sec_t, cap=0.25)
                except Exception:
                    pass

            # ---- Realized next-period portfolio excess return ----
            ret = float((w_ser * y_t.droplevel(0).reindex(w_ser.index)).sum())

            # ---- Turnover (L1 change) ----
            if last_w is None:
                turn = float(w_ser.abs().sum())
            else:
                u = w_ser.index.union(last_w.index)
                turn = float((w_ser.reindex(u, fill_value=0.0) - last_w.reindex(u, fill_value=0.0)).abs().sum())
            last_w = w_ser

            ret_list.append(_pd.Series({t: ret}))
            turn_list.append(_pd.Series({t: turn}))

    # ---- Collect series ----
    returns = _pd.concat(ret_list).sort_index() if ret_list else _pd.Series(dtype=float)
    turnovers = _pd.concat(turn_list).sort_index() if turn_list else _pd.Series(0.0, index=returns.index)

    # ---- Apply linear costs ----
    if len(returns) and len(turnovers):
        ret_net_arr = apply_trading_costs(returns.values, turnovers.values, cost_bps)
        returns_net = _pd.Series(ret_net_arr, index=returns.index)
    else:
        returns_net = returns

    # ---- Stats ----
    freq = _freq_from_days(rebalance_days)
    stats = perf_summary(returns_net if len(returns_net) else _pd.Series([0.0]), freq=freq)

    # ---- IC summary ----
    ic_mean = float(_np.nanmean(ics)) if ics else float("nan")
    ic_ir = float(_np.nanmean(ics) / (_np.nanstd(ics) + 1e-12)) if ics else float("nan")

    return {"ic_mean": ic_mean, "ic_ir": ic_ir, "stats": stats, "returns_net": returns_net}
