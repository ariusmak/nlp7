from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.paths import paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    processed = p["data"]["processed"]

    posts = pd.read_parquet(processed / "posts_scored.parquet")
    events = pd.read_parquet(processed / "event_sentiment_features.parquet")

    pp = posts.groupby("ticker").agg(
        linked_posts=("id", "size"),
        unique_posts=("id", "nunique"),
        multi_share=("multi_ticker_flag", "mean"),
        mean_compound=("adp_compound", "mean"),
        median_compound=("adp_compound", "median"),
        pct_pos=("adp_label", lambda s: (s == "positive").mean()),
        pct_neg=("adp_label", lambda s: (s == "negative").mean()),
    )

    ev = events.groupby("ticker").agg(
        events=("event_id", "size"),
        zero=("post_count", lambda s: int((s == 0).sum())),
        thin=("post_count", lambda s: int((s < 10).sum())),
        posts_mean=("post_count", "mean"),
        posts_min=("post_count", "min"),
        posts_med=("post_count", "median"),
        posts_max=("post_count", "max"),
        ev_mean_sent=("mean_sentiment", "mean"),
    )

    tab = ev.join(pp).sort_values("posts_mean", ascending=False)
    tab["multi_share"] = tab["multi_share"].round(3)
    tab["mean_compound"] = tab["mean_compound"].round(3)
    tab["median_compound"] = tab["median_compound"].round(3)
    tab["pct_pos"] = (tab["pct_pos"] * 100).round(1)
    tab["pct_neg"] = (tab["pct_neg"] * 100).round(1)
    tab["posts_mean"] = tab["posts_mean"].round(1)
    tab["ev_mean_sent"] = tab["ev_mean_sent"].round(3)

    cols = ["events", "zero", "thin", "posts_mean", "posts_med", "posts_min", "posts_max",
            "unique_posts", "linked_posts", "multi_share", "ev_mean_sent", "mean_compound",
            "pct_pos", "pct_neg"]
    print(tab[cols].to_string())
    print()
    print("totals: events={}, linked_posts={:,}, unique_posts_linked={:,}".format(
        int(tab["events"].sum()),
        int(posts.shape[0]),
        int(posts["id"].nunique()),
    ))

    out = p["outputs"]["tables"] / "per_stock_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab[cols].to_csv(out)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
