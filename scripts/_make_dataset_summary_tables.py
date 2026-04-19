from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_parquet(BASE / "data/raw/reddit/p4p_reddit_posts.parquet")
ts = pd.to_datetime(raw["created_utc"], unit="s", errors="coerce")
in_window = ts.between(pd.Timestamp("2016-01-01"), pd.Timestamp("2023-03-29"))

rb = raw["selftext"].astype(str).str.strip()
rb_valid = (~rb.isin(["", "[removed]", "[deleted]", "nan", "None"])) & rb.notna()

long_match = pd.read_parquet(BASE / "data/interim/posts_matched_long.parquet")
linked = pd.read_parquet(BASE / "data/interim/posts_linked_to_events.parquet")
events = pd.read_parquet(BASE / "data/processed/event_modeling_dataset.parquet")

summary_rows = [
    ("Raw WSB posts in p4p dump",            len(raw),                      "title + selftext, 2012-04-11 → 2023-03-28"),
    ("Posts with real body text",            int(rb_valid.sum()),           "selftext not empty / [removed] / [deleted]"),
    ("Posts in safe window (2016 → 2023-03-28)", int(in_window.sum()),       "window used for all downstream matching"),
    ("Real-body posts in safe window",       int((rb_valid & in_window).sum()), ""),
    ("Posts matched to ≥1 ticker (global)",  long_match["id"].nunique(),    "cashtag / alias / ticker+cue hierarchy"),
    ("Post-ticker match rows",               len(long_match),               "one row per (post, matched ticker)"),
    ("Multi-ticker posts (≥2 tickers)",      int((long_match.drop_duplicates("id")["multi_ticker_flag"] == 1).sum()), "share: {:.1%}".format((long_match.drop_duplicates("id")["multi_ticker_flag"] == 1).mean())),
    ("Posts linked to an earnings window",   linked["id"].nunique(),        "10-day pre-earnings window, excl. rdq"),
    ("Post-event link rows",                 len(linked),                   "one row per (post, earnings event)"),
    ("Real-body post-event links",           int(linked["has_real_body"].sum()), ""),
    ("Earnings events in final dataset",     len(events),                   "15 tickers × 29 quarters, 2016 → 2023-Q1"),
    ("Events with ≥1 matched post",          int((events["post_count"] >= 1).sum()), "{:.1%} of events".format((events["post_count"] >= 1).mean())),
    ("Events with thin coverage (<10 posts)", int((events["post_count"] < 10).sum()), "{:.1%} of events".format((events["post_count"] < 10).mean())),
]
summary = pd.DataFrame(summary_rows, columns=["metric", "count", "note"])
summary.to_csv(OUT / "dataset_summary_totals.csv", index=False)

global_per_stock = (
    long_match.groupby("ticker")
    .agg(
        unique_posts=("id", "nunique"),
        post_ticker_rows=("id", "size"),
        real_body_posts=(
            "selftext",
            lambda s: (
                (~s.astype(str).str.strip().isin(["", "[removed]", "[deleted]", "nan", "None"]))
                & s.notna()
            ).sum(),
        ),
    )
    .reset_index()
)
multi_by_tkr = (
    long_match[long_match["multi_ticker_flag"] == 1]
    .groupby("ticker")["id"].nunique().rename("multi_ticker_posts")
)
global_per_stock = global_per_stock.merge(multi_by_tkr, on="ticker", how="left").fillna({"multi_ticker_posts": 0})
global_per_stock["multi_ticker_posts"] = global_per_stock["multi_ticker_posts"].astype(int)
global_per_stock["multi_ticker_share"] = (
    global_per_stock["multi_ticker_posts"] / global_per_stock["unique_posts"]
).round(3)
global_per_stock["real_body_share"] = (
    global_per_stock["real_body_posts"] / global_per_stock["unique_posts"]
).round(3)
global_per_stock = global_per_stock.sort_values("unique_posts", ascending=False).reset_index(drop=True)
global_per_stock.to_csv(OUT / "dataset_summary_matched_per_stock.csv", index=False)

per_event = linked.groupby(["ticker", "event_id"]).size().rename("posts_per_event").reset_index()
window_stats = (
    per_event.groupby("ticker")["posts_per_event"]
    .agg(
        events="size",
        post_event_links="sum",
        mean_posts="mean",
        median_posts="median",
        min_posts="min",
        max_posts="max",
    )
    .reset_index()
)

all_events_per_ticker = events.groupby("ticker").size().rename("events_total").reset_index()
zero_events = (
    events[events["post_count"] == 0].groupby("ticker").size().rename("events_zero_posts").reset_index()
)
thin_events = (
    events[events["post_count"] < 10].groupby("ticker").size().rename("events_thin_lt10").reset_index()
)

real_body_links = (
    linked.groupby("ticker")["has_real_body"].sum().rename("real_body_links").reset_index()
)

window_stats = (
    all_events_per_ticker
    .merge(window_stats.drop(columns="events"), on="ticker", how="left")
    .merge(zero_events, on="ticker", how="left")
    .merge(thin_events, on="ticker", how="left")
    .merge(real_body_links, on="ticker", how="left")
    .fillna(0)
)

for col in [
    "post_event_links", "min_posts", "max_posts",
    "events_zero_posts", "events_thin_lt10", "real_body_links",
]:
    window_stats[col] = window_stats[col].astype(int)

window_stats["mean_posts"] = window_stats["mean_posts"].round(1)
window_stats["median_posts"] = window_stats["median_posts"].round(1)

window_stats = window_stats.sort_values("post_event_links", ascending=False).reset_index(drop=True)
window_stats = window_stats[[
    "ticker", "events_total", "post_event_links", "real_body_links",
    "mean_posts", "median_posts", "min_posts", "max_posts",
    "events_zero_posts", "events_thin_lt10",
]]
window_stats.to_csv(OUT / "dataset_summary_window_per_stock.csv", index=False)

print("Wrote:")
print(" -", OUT / "dataset_summary_totals.csv")
print(" -", OUT / "dataset_summary_matched_per_stock.csv")
print(" -", OUT / "dataset_summary_window_per_stock.csv")
print()
print("=== Totals ===")
print(summary.to_string(index=False))
print()
print("=== Globally matched posts per stock ===")
print(global_per_stock.to_string(index=False))
print()
print("=== Posts per stock within 10-day pre-earnings windows ===")
print(window_stats.to_string(index=False))
