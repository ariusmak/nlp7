from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.paths import paths, universe
from utils.wrds_io import load_earnings_dates


def main() -> None:
    p = paths()
    safe = universe()["safe_window"]
    start = pd.Timestamp(safe["start"])
    end = pd.Timestamp(safe["end"])

    df, source = load_earnings_dates()
    df = df[(df["rdq"] >= start) & (df["rdq"] <= end)].reset_index(drop=True)

    tickers = [row["ticker"] for row in universe()["tickers"]]
    df = df[df["ticker"].isin(tickers)].reset_index(drop=True)

    out_csv = p["files"]["earnings_dates_csv"]
    out_parq = p["files"]["earnings_dates_parquet"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parq, index=False)

    print(f"source           : {source}")
    print(f"earnings events  : {len(df):,}")
    print(f"tickers covered  : {df['ticker'].nunique()} / {len(tickers)}")
    print(f"date range       : {df['rdq'].min().date()}  ->  {df['rdq'].max().date()}")
    print()
    print("events per ticker:")
    print(df.groupby("ticker").size().sort_index().to_string())
    print()
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_parq}")


if __name__ == "__main__":
    main()
