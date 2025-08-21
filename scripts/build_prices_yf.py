# scripts/build_prices_yf.py
from pathlib import Path
import pandas as pd
import yfinance as yf

# Include SPY (benchmark) + a small, liquid universe to start
TICKERS = [
    "SPY","AAPL","MSFT","AMZN","GOOGL","META","NVDA","JPM","XOM","UNH",
    "PG","HD","V","MA","PFE","KO","PEP","CVX","BAC","CSCO"
]

def main():
    out_path = Path("data/raw/stocks.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        tickers=" ".join(TICKERS),
        period="10y",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        interval="1d"
    )

    # Build MultiIndex (date, symbol) with minimal cols: close, volume
    frames = []
    for t in TICKERS:
        if t not in df.columns.levels[0]:
            continue
        sub = df[t][["Close","Volume"]].rename(columns={"Close":"close","Volume":"volume"})
        sub["symbol"] = t
        frames.append(sub.reset_index())

    wide = pd.concat(frames, ignore_index=True)
    wide = wide.rename(columns={"Date":"date"})
    wide = wide.dropna(subset=["close"])

    multi = wide.set_index(["date","symbol"]).sort_index()
    multi.to_parquet(out_path)
    print(f"✅ Wrote {out_path}  ({multi.index.get_level_values(1).nunique()} symbols, {multi.index.get_level_values(0).nunique()} dates)")

if __name__ == "__main__":
    main()
