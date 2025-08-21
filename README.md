# swing-backtester - For backtesting swing trades (1wk - 1mo time horizon)

# swing-backtester


Backtesting framework for **swing strategies** (1 week – 1 month) with **XGBoost** and optional **sentiment** features. Profit‑first defaults, purged walk‑forward CV, realistic costs.


## Quickstart
```bash
python -m pip install -r requirements.txt
python scripts/run.py --horizon weekly # or: daily, weekly, monthly

Data: Place daily OHLCV parquet/csv under data/raw/ or point the loader to your store (see config/params.yaml).

Structure

src/swingbt/features: feature engineering (momentum, drawdown, vol, RSI, etc.)

src/swingbt/targets: horizon‑aligned label generation

src/swingbt/models: XGBoost training & scoring

src/swingbt/backtest: walk‑forward loop + metrics

src/swingbt/utils: loaders, costs, time split, portfolio construction

Outputs

Backtest results (equity curve, IC stats, CSVs) are saved to out/.