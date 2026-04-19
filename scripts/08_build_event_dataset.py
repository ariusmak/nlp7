from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from utils.paths import paths


SENTIMENT_COLS = ["post_count",
                  "mean_sentiment", "median_sentiment", "std_sentiment",
                  "frac_positive", "frac_negative"]
SENTIMENT_RAW_COLS = ["mean_sentiment_raw", "median_sentiment_raw", "std_sentiment_raw",
                      "frac_positive_raw", "frac_negative_raw"]
MARKET_COLS = ["ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    processed = p["data"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    sent = pd.read_parquet(processed / "event_sentiment_features.parquet")
    mkt = pd.read_parquet(processed / "event_market_features.parquet")
    earnings = pd.read_parquet(p["files"]["earnings_dates_parquet"])
    earnings["rdq"] = pd.to_datetime(earnings["rdq"]).dt.normalize()
    earnings["event_id"] = earnings["ticker"].astype(str) + "_" + earnings["rdq"].dt.strftime("%Y%m%d")
    earnings["calendar_year"] = earnings["rdq"].dt.year

    meta = earnings[["ticker", "event_id", "rdq", "fyearq", "fqtr",
                     "earnings_timing", "timing_source", "calendar_year"]].drop_duplicates("event_id")

    df = meta.merge(mkt.drop(columns=["ticker", "rdq"], errors="ignore"), on="event_id", how="left")
    df = df.merge(sent.drop(columns=["ticker", "rdq"], errors="ignore"), on="event_id", how="left")

    df["has_sentiment"] = df["post_count"].fillna(0) > 0
    df["post_count"] = df["post_count"].fillna(0).astype(int)
    df["thin_sentiment"] = df["post_count"] < 10

    out = processed / "event_modeling_dataset.parquet"
    df.to_parquet(out, index=False)

    print(f"rows                          : {len(df):,}")
    print(f"tickers                       : {df['ticker'].nunique()}")
    print(f"calendar years                : {sorted(df['calendar_year'].unique().tolist())}")
    print()

    print("coverage:")
    print(f"  events with target          : {df['target_ret3'].notna().sum():,}")
    print(f"  events with all 4 mkt feats : {df[MARKET_COLS].notna().all(axis=1).sum():,}")
    print(f"  events with >=1 post        : {int(df['has_sentiment'].sum()):,}")
    print(f"  events with >=10 posts      : {int((df['post_count'] >= 10).sum()):,}")
    print(f"  zero-post events            : {int((df['post_count'] == 0).sum()):,}")
    print()

    train_years = list(range(2016, 2022))
    test_years = [2022, 2023]
    n_train = int(df["calendar_year"].isin(train_years).sum())
    n_test = int(df["calendar_year"].isin(test_years).sum())
    print(f"train (2016-2021) events      : {n_train:,}")
    print(f"test  (2022-2023) events      : {n_test:,}")
    print()

    print("events per (ticker, year):")
    by_yr = df.groupby(["calendar_year", "ticker"]).size().unstack(fill_value=0)
    print(by_yr.to_string())
    print()

    print("feature null-share:")
    for c in MARKET_COLS + SENTIMENT_COLS + ["target_ret3"]:
        n_null = int(df[c].isna().sum())
        print(f"  {c:>20}  nulls={n_null:>4}  ({n_null/len(df)*100:.1f}%)")

    tables_dir = p["outputs"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    by_yr.to_csv(tables_dir / "events_per_ticker_year.csv")
    print()
    print(f"wrote: {out}")
    print(f"wrote: {tables_dir / 'events_per_ticker_year.csv'}")


if __name__ == "__main__":
    main()
