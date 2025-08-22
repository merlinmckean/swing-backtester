# scripts/build_universe_sp500.py
from pathlib import Path
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def _to_yahoo_symbol(t: str) -> str:
    # Yahoo expects dots as dashes, e.g., BRK.B -> BRK-B
    return t.replace(".", "-").strip()

def main():
    out_dir = Path("data/universe")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sp500.constituents.csv"

    # Pull tables; the first table on this page is the constituents list
    tables = pd.read_html(WIKI_URL)
    df = tables[0].copy()

    # Standardize columns we care about (Wikipedia headers can vary slightly)
    # Common columns: 'Symbol', 'Security', 'GICS Sector'
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol") or "Symbol"
    name_col = cols.get("security") or "Security"
    sector_col = next((c for c in df.columns if "GICS" in c and "Sector" in c), "GICS Sector")

    df = df.rename(columns={sym_col: "symbol_raw", name_col: "name", sector_col: "sector"})
    df["symbol"] = df["symbol_raw"].astype(str).map(_to_yahoo_symbol)
    df = df[["symbol", "name", "sector"]].dropna().drop_duplicates().sort_values("symbol")

    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}  ({len(df)} symbols)")

if __name__ == "__main__":
    main()
