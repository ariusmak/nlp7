from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from utils.paths import paths, universe
from utils.wrds_io import pull_crsp_daily_from_wrds


DAILY_START = "2015-10-01"
DAILY_END = "2023-04-28"


def load_or_pull_daily(p: dict) -> tuple[pd.DataFrame, str]:
    out = p["files"]["crsp_daily_parquet"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        df = pd.read_parquet(out)
        return df, "file"
    tickers = [row["ticker"] for row in universe()["tickers"]]
    df = pull_crsp_daily_from_wrds(DAILY_START, DAILY_END, tickers)
    df.to_parquet(out, index=False)
    return df, "wrds"


def _trading_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["tidx"] = df.groupby("ticker").cumcount()
    return df


def _pre_event_row(ticker_df: pd.DataFrame, rdq: pd.Timestamp) -> int | None:
    idx = ticker_df.index[ticker_df["date"] < rdq]
    if len(idx) == 0:
        return None
    return int(idx[-1])


def _nth_trading_day_on_or_after(ticker_df: pd.DataFrame, rdq: pd.Timestamp, n: int) -> int | None:
    mask = ticker_df["date"] >= rdq
    if not mask.any():
        return None
    first = int(ticker_df.index[mask][0])
    target = first + n
    if target >= len(ticker_df):
        return None
    return target


def compute_features_for_event(ticker_df: pd.DataFrame, rdq: pd.Timestamp) -> dict:
    out = {
        "ret5_pre": np.nan,
        "ret20_pre": np.nan,
        "vol20_pre": np.nan,
        "abvol_pre": np.nan,
        "target_ret3": np.nan,
        "target_start_date": pd.NaT,
        "target_end_date": pd.NaT,
        "timing_used": "fallback_prev_close",
    }

    last_pre = _pre_event_row(ticker_df, rdq)
    if last_pre is None:
        return out

    pre_returns_20 = ticker_df["ret"].iloc[max(0, last_pre - 19): last_pre + 1]
    if len(pre_returns_20) == 20:
        r20 = (1.0 + pre_returns_20).prod() - 1.0
        out["ret20_pre"] = float(r20)
        out["vol20_pre"] = float(pre_returns_20.std(ddof=1))
    pre_returns_5 = ticker_df["ret"].iloc[max(0, last_pre - 4): last_pre + 1]
    if len(pre_returns_5) == 5:
        out["ret5_pre"] = float((1.0 + pre_returns_5).prod() - 1.0)

    vol_dminus1 = ticker_df["vol"].iloc[last_pre]
    vol_window = ticker_df["vol"].iloc[max(0, last_pre - 20): last_pre]
    if len(vol_window) == 20 and vol_window.mean() > 0:
        out["abvol_pre"] = float(vol_dminus1 / vol_window.mean())

    t0_price = float(ticker_df["prc"].iloc[last_pre])
    t3_idx = last_pre + 3
    if t3_idx < len(ticker_df):
        t3_price = float(ticker_df["prc"].iloc[t3_idx])
        if t0_price > 0:
            out["target_ret3"] = (t3_price / t0_price) - 1.0
            out["target_start_date"] = ticker_df["date"].iloc[last_pre]
            out["target_end_date"] = ticker_df["date"].iloc[t3_idx]

    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    processed = p["data"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    daily, source = load_or_pull_daily(p)
    print(f"daily market data source : {source}")
    print(f"daily rows               : {len(daily):,}")
    print(f"daily tickers            : {daily['ticker'].nunique()}")
    print(f"daily date range         : {daily['date'].min().date()} -> {daily['date'].max().date()}")
    print()

    earnings = pd.read_parquet(p["files"]["earnings_dates_parquet"])
    earnings["rdq"] = pd.to_datetime(earnings["rdq"]).dt.normalize()
    earnings["event_id"] = earnings["ticker"].astype(str) + "_" + earnings["rdq"].dt.strftime("%Y%m%d")

    by_ticker: dict[str, pd.DataFrame] = {
        t: df.reset_index(drop=True) for t, df in daily.groupby("ticker")
    }

    rows = []
    for _, ev in earnings.iterrows():
        t = ev["ticker"]
        rdq = ev["rdq"]
        tdf = by_ticker.get(t)
        if tdf is None:
            feats = {k: np.nan for k in ["ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre", "target_ret3"]}
            feats.update({"target_start_date": pd.NaT, "target_end_date": pd.NaT, "timing_used": "no_data"})
        else:
            feats = compute_features_for_event(tdf, rdq)
        feats.update({"ticker": t, "event_id": ev["event_id"], "rdq": rdq})
        rows.append(feats)

    feats_df = pd.DataFrame(rows)
    keep = ["ticker", "event_id", "rdq",
            "ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre",
            "target_ret3", "target_start_date", "target_end_date", "timing_used"]
    feats_df = feats_df[keep]

    out = processed / "event_market_features.parquet"
    feats_df.to_parquet(out, index=False)

    n = len(feats_df)
    n_target = int(feats_df["target_ret3"].notna().sum())
    n_feats = int(feats_df[["ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre"]].notna().all(axis=1).sum())
    print(f"events                   : {n:,}")
    print(f"events with target       : {n_target:,}")
    print(f"events with all 4 feats  : {n_feats:,}")
    print()

    print("feature distributions:")
    desc = feats_df[["ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre", "target_ret3"]].describe().round(4)
    print(desc.to_string())
    print()

    per_tkr = (
        feats_df.groupby("ticker")
        .agg(events=("event_id", "size"),
             target_mean=("target_ret3", "mean"),
             target_std=("target_ret3", "std"),
             target_pos_frac=("target_ret3", lambda s: float((s > 0).mean())))
        .round(4)
        .sort_values("target_std", ascending=False)
    )
    print("per-ticker target (3-day post-earnings return):")
    print(per_tkr.to_string())

    tables_dir = p["outputs"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    desc.to_csv(tables_dir / "market_feature_distributions.csv")
    per_tkr.to_csv(tables_dir / "market_target_per_ticker.csv")
    print()
    print(f"wrote: {out}")
    print(f"wrote: {tables_dir / 'market_feature_distributions.csv'}")
    print(f"wrote: {tables_dir / 'market_target_per_ticker.csv'}")


if __name__ == "__main__":
    main()
