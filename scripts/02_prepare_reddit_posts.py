from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.paths import paths, sentiment_config


PLACEHOLDERS = {"[removed]", "[deleted]"}


def main() -> None:
    p = paths()
    cfg = sentiment_config()
    window_days = int(cfg["event_window_days"])

    posts_path = p["files"]["reddit_posts_parquet"]
    earnings_path = p["files"]["earnings_dates_parquet"]

    earnings = pd.read_parquet(earnings_path)
    earnings["rdq"] = pd.to_datetime(earnings["rdq"]).dt.normalize()
    earnings["window_start"] = earnings["rdq"] - pd.Timedelta(days=window_days)

    window_dates = set()
    for _, r in earnings.iterrows():
        window_dates.update(
            d.date() for d in pd.date_range(r["window_start"], r["rdq"] - pd.Timedelta(days=1), freq="D")
        )

    posts = pd.read_parquet(posts_path)
    n_raw = len(posts)

    posts["created_utc"] = pd.to_datetime(posts["created_utc"], errors="coerce")
    posts = posts.dropna(subset=["created_utc"]).reset_index(drop=True)
    posts["post_date"] = posts["created_utc"].dt.date

    posts = posts[posts["post_date"].isin(window_dates)].reset_index(drop=True)
    n_after_window = len(posts)

    posts["title"] = posts["title"].fillna("").astype(str)
    posts["selftext"] = posts["selftext"].fillna("").astype(str)

    title_empty = posts["title"].str.strip().eq("")
    n_empty_title = int(title_empty.sum())
    posts = posts[~title_empty].reset_index(drop=True)
    n_kept = len(posts)

    body_stripped = posts["selftext"].str.strip()
    is_placeholder = body_stripped.isin(PLACEHOLDERS)
    has_real_body = (~is_placeholder) & (body_stripped.str.len() >= 20)
    n_with_body = int(has_real_body.sum())
    n_placeholder_body = int(is_placeholder.sum())

    counts_per_window = []
    post_dates = pd.to_datetime(posts["post_date"])
    for _, r in earnings.iterrows():
        ws = r["window_start"]
        we = r["rdq"] - pd.Timedelta(days=1)
        c = int(((post_dates >= ws) & (post_dates <= we)).sum())
        counts_per_window.append(c)
    counts = pd.Series(counts_per_window)

    interim = p["data"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    out = interim / "reddit_posts_in_windows.parquet"
    posts.to_parquet(out, index=False)

    print(f"raw rows                      : {n_raw:,}")
    print(f"after window filter           : {n_after_window:,}  (dropped {n_raw - n_after_window:,})")
    print(f"dropped: empty title          : {n_empty_title:,}")
    print(f"kept (final)                  : {n_kept:,}")
    print()
    print(f"kept with real body (>=20ch)  : {n_with_body:,}  ({n_with_body/max(n_kept,1)*100:.1f}%)")
    print(f"kept with placeholder body    : {n_placeholder_body:,}  (retained per policy)")
    print()
    print(f"earnings events               : {len(earnings):,}")
    print(f"mean posts / window (pre-match): {counts.mean():,.1f}")
    print(f"median posts / window          : {counts.median():,.0f}")
    print(f"min / max posts / window       : {counts.min():,} / {counts.max():,}")
    print()
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
