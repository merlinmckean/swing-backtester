# scripts/build_prices_yf.py
from pathlib import Path
import pandas as pd
import yfinance as yf

UNIVERSE_CSV = Path("data/universe/sp500.constituents.csv")
OUT_PARQUET = Path("data/raw/stocks.parquet")

# Always include SPY for labeling/benchmark
ALWAYS_INCLUDE = ["SPY"]

def main():
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    tickers = []
    sector_map = {}

    if UNIVERSE_CSV.exists():
        uni = pd.read_csv(UNIVERSE_CSV)
        tickers = sorted(set(uni["symbol"].astype(str)) | set(ALWAYS_INCLUDE))
        sector_map = dict(zip(uni["symbol"].astype(str), uni["sector"].astype(str)))
        print(f"Using universe CSV with {len(tickers)} tickers (incl SPY).")
    else:
        print("⚠️ Universe CSV not found, falling back to small starter list.")
        tickers = [
            "SPY","AAPL","MSFT","AMZN","GOOGL","META","NVDA","JPM","XOM","UNH",
            "PG","HD","V","MA","PFE","KO","PEP","CVX","BAC","CSCO"
        ]
        sector_map = {}

    # Download daily OHLCV
    df = yf.download(
        tickers,
        period="10y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    # yfinance returns a wide df, one top-level column per ticker
    frames = []
    have = set()
    for sym in tickers:
        try:
            sub = df[sym].copy()
        except Exception:
            continue
        if sub.empty:
            continue
        sub.columns = [c.lower() for c in sub.columns]
        if "adj close" in sub.columns:
            sub = sub.rename(columns={"adj close": "adj_close"})
        sub["symbol"] = sym
        frames.append(sub.reset_index())
        have.add(sym)

    if not frames:
        raise RuntimeError("No data frames downloaded — check ticker list/network.")

    long = pd.concat(frames, ignore_index=True)

    # Minimal schema expected downstream
    keep_cols = ["Date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    long = long[[c for c in keep_cols if c in long.columns]].rename(columns={"Date": "date"})

    # Sector column (optional)
    long["sector"] = long["symbol"].map(sector_map)

    # MultiIndex (date, symbol)
    long = long.set_index(["date", "symbol"]).sort_index()

    long.to_parquet(OUT_PARQUET)
    print(f"✅ Wrote {OUT_PARQUET}  ({long.index.get_level_values(1).nunique()} symbols, "
          f"{long.index.get_level_values(0).nunique()} dates)")
    missing = sorted(set(tickers) - have)
    if missing:
        print(f"Note: {len(missing)} symbols had no data and were skipped (common: recent IPOs, renamed tickers).")

if __name__ == "__main__":
    main()
