from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from utils.paths import paths, sentiment_config


def _aggregate(group: pd.DataFrame, compound_col: str, pos_thr: float, neg_thr: float) -> dict:
    c = group[compound_col].to_numpy()
    n = c.size
    if n == 0:
        return {
            "post_count": 0,
            "mean_sentiment": np.nan,
            "median_sentiment": np.nan,
            "std_sentiment": np.nan,
            "frac_positive": np.nan,
            "frac_negative": np.nan,
        }
    return {
        "post_count": int(n),
        "mean_sentiment": float(np.mean(c)),
        "median_sentiment": float(np.median(c)),
        "std_sentiment": float(np.std(c, ddof=1)) if n > 1 else 0.0,
        "frac_positive": float(np.mean(c > pos_thr)),
        "frac_negative": float(np.mean(c < neg_thr)),
    }


def build_event_features(
    posts: pd.DataFrame,
    events: pd.DataFrame,
    compound_col: str,
    pos_thr: float,
    neg_thr: float,
    suffix: str = "",
) -> pd.DataFrame:
    rows = []
    for (ticker, event_id, rdq), grp in posts.groupby(["ticker", "event_id", "rdq"], sort=False):
        feats = _aggregate(grp, compound_col, pos_thr, neg_thr)
        feats.update({"ticker": ticker, "event_id": event_id, "rdq": rdq})
        rows.append(feats)

    if rows:
        agg = pd.DataFrame(rows)
    else:
        agg = pd.DataFrame(columns=["ticker", "event_id", "rdq",
                                    "post_count", "mean_sentiment", "median_sentiment",
                                    "std_sentiment", "frac_positive", "frac_negative"])

    full = events.merge(agg, on=["ticker", "event_id", "rdq"], how="left")
    full["post_count"] = full["post_count"].fillna(0).astype(int)
    if suffix:
        rename = {c: c + suffix for c in
                  ["mean_sentiment", "median_sentiment", "std_sentiment",
                   "frac_positive", "frac_negative"]}
        full = full.rename(columns=rename)
    return full


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    cfg = sentiment_config()
    pos_thr = float(cfg["compound_thresholds"]["positive"])
    neg_thr = float(cfg["compound_thresholds"]["negative"])

    processed = p["data"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    posts = pd.read_parquet(processed / "posts_scored.parquet")
    earnings = pd.read_parquet(p["files"]["earnings_dates_parquet"])
    earnings["rdq"] = pd.to_datetime(earnings["rdq"]).dt.normalize()
    events = earnings[["ticker", "rdq"]].copy()
    events["event_id"] = events["ticker"].astype(str) + "_" + events["rdq"].dt.strftime("%Y%m%d")
    events = events[["ticker", "event_id", "rdq"]].drop_duplicates().reset_index(drop=True)

    posts["rdq"] = pd.to_datetime(posts["rdq"]).dt.normalize()
    posts = posts.dropna(subset=["adp_compound", "raw_compound"])
    print(f"loaded posts_scored: {len(posts):,} rows")
    print(f"earnings events    : {len(events):,}")

    adapted = build_event_features(posts, events, "adp_compound", pos_thr, neg_thr, suffix="")
    with_raw = build_event_features(posts, events, "raw_compound", pos_thr, neg_thr, suffix="_raw")

    keep_raw_cols = ["ticker", "event_id", "rdq",
                     "mean_sentiment_raw", "median_sentiment_raw", "std_sentiment_raw",
                     "frac_positive_raw", "frac_negative_raw"]
    combined = adapted.merge(with_raw[keep_raw_cols], on=["ticker", "event_id", "rdq"], how="left")

    out = processed / "event_sentiment_features.parquet"
    combined.to_parquet(out, index=False)

    n_total = len(combined)
    n_with = int((combined["post_count"] > 0).sum())
    n_zero = int((combined["post_count"] == 0).sum())
    print(f"wrote: {out}  ({n_total:,} events; {n_with:,} with posts, {n_zero} zero)")
    print()

    nonzero = combined[combined["post_count"] > 0]
    summary = pd.DataFrame({
        "feature": ["post_count", "mean_sentiment", "median_sentiment",
                    "std_sentiment", "frac_positive", "frac_negative"],
        "mean": [
            nonzero["post_count"].mean(),
            nonzero["mean_sentiment"].mean(),
            nonzero["median_sentiment"].mean(),
            nonzero["std_sentiment"].mean(),
            nonzero["frac_positive"].mean(),
            nonzero["frac_negative"].mean(),
        ],
        "median": [
            nonzero["post_count"].median(),
            nonzero["mean_sentiment"].median(),
            nonzero["median_sentiment"].median(),
            nonzero["std_sentiment"].median(),
            nonzero["frac_positive"].median(),
            nonzero["frac_negative"].median(),
        ],
        "std": [
            nonzero["post_count"].std(),
            nonzero["mean_sentiment"].std(),
            nonzero["median_sentiment"].std(),
            nonzero["std_sentiment"].std(),
            nonzero["frac_positive"].std(),
            nonzero["frac_negative"].std(),
        ],
    })
    print("event-level feature summary (adapted, N={:,} events with >=1 post):".format(len(nonzero)))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    per_ticker = (
        combined.groupby("ticker")
        .agg(events=("event_id", "size"),
             zero_events=("post_count", lambda s: int((s == 0).sum())),
             thin_events=("post_count", lambda s: int((s < 10).sum())),
             mean_posts=("post_count", "mean"),
             mean_sent=("mean_sentiment", "mean"),
             mean_fracpos=("frac_positive", "mean"),
             mean_fracneg=("frac_negative", "mean"))
        .sort_values("mean_posts", ascending=False)
    )
    per_ticker["mean_posts"] = per_ticker["mean_posts"].round(1)
    per_ticker["mean_sent"] = per_ticker["mean_sent"].round(4)
    per_ticker["mean_fracpos"] = per_ticker["mean_fracpos"].round(4)
    per_ticker["mean_fracneg"] = per_ticker["mean_fracneg"].round(4)
    print("per-ticker event-level sentiment (adapted):")
    print(per_ticker.to_string())
    print()

    tables_dir = p["outputs"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "event_sentiment_summary.csv", index=False)
    per_ticker.to_csv(tables_dir / "event_sentiment_per_ticker.csv")
    print(f"wrote: {tables_dir / 'event_sentiment_summary.csv'}")
    print(f"wrote: {tables_dir / 'event_sentiment_per_ticker.csv'}")


if __name__ == "__main__":
    main()
