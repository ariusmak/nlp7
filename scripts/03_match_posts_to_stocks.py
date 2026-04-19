from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from utils.paths import paths
from utils.stock_matcher import StockMatcher


def main() -> None:
    p = paths()
    interim = p["data"]["interim"]
    in_posts = interim / "reddit_posts_in_windows.parquet"

    posts = pd.read_parquet(in_posts)
    n = len(posts)

    matcher = StockMatcher.build()
    title = posts["title"].fillna("").astype(str)
    body = posts["selftext"].fillna("").astype(str) if "selftext" in posts.columns else pd.Series([""] * len(posts))
    text = (title + " " + body).astype(str)
    tqdm.pandas(desc="matching tickers", mininterval=2.0)
    matches = text.progress_apply(matcher.match)

    posts["matched_tickers"] = matches
    posts["num_matched"] = matches.map(len)
    posts["multi_ticker_flag"] = (posts["num_matched"] > 1).astype(int)

    wide = posts.copy()
    wide["matched_tickers"] = wide["matched_tickers"].map(lambda xs: ",".join(xs))
    wide_out = interim / "posts_matched_wide.parquet"
    wide.to_parquet(wide_out, index=False)

    long = posts[posts["num_matched"] > 0].copy()
    long = long.explode("matched_tickers", ignore_index=True)
    long = long.rename(columns={"matched_tickers": "ticker"})
    long_out = interim / "posts_matched_long.parquet"
    long.to_parquet(long_out, index=False)

    n_any = int((posts["num_matched"] > 0).sum())
    n_multi = int((posts["num_matched"] > 1).sum())

    print(f"posts scanned                : {n:,}")
    print(f"posts with >=1 match          : {n_any:,}  ({n_any/n*100:.1f}%)")
    print(f"posts with multi-ticker flag  : {n_multi:,}  ({n_multi/n*100:.1f}%)")
    print()
    print("matched-post counts per ticker (post-level, counts multi-ticker posts in each):")
    per_ticker = long.groupby("ticker").size().sort_values(ascending=False)
    print(per_ticker.to_string())
    print()
    print(f"wrote: {wide_out}   ({len(wide):,} rows)")
    print(f"wrote: {long_out}   ({len(long):,} rows)")


if __name__ == "__main__":
    main()
