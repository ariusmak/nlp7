from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.paths import paths, sentiment_config


PLACEHOLDERS = {"[removed]", "[deleted]"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    cfg = sentiment_config()
    window_days = int(cfg["event_window_days"])

    interim = p["data"]["interim"]
    earnings = pd.read_parquet(p["files"]["earnings_dates_parquet"])
    earnings["rdq"] = pd.to_datetime(earnings["rdq"]).dt.normalize()
    earnings["window_start"] = earnings["rdq"] - pd.Timedelta(days=window_days)
    earnings["event_id"] = (
        earnings["ticker"].astype(str) + "_" + earnings["rdq"].dt.strftime("%Y%m%d")
    )

    posts_long = pd.read_parquet(interim / "posts_matched_long.parquet")
    posts_long["created_utc"] = pd.to_datetime(posts_long["created_utc"], errors="coerce")
    posts_long = posts_long.dropna(subset=["created_utc"]).reset_index(drop=True)
    posts_long["post_dt"] = posts_long["created_utc"].dt.normalize()

    body_stripped = posts_long["selftext"].fillna("").astype(str).str.strip()
    posts_long["has_real_body"] = (~body_stripped.isin(PLACEHOLDERS)) & (body_stripped.str.len() >= 20)

    links = posts_long.merge(earnings, on="ticker", how="inner", suffixes=("", "_evt"))
    mask = (links["post_dt"] >= links["window_start"]) & (links["post_dt"] < links["rdq"])
    links = links[mask].reset_index(drop=True)

    keep_cols = [
        "id", "ticker", "event_id", "rdq", "window_start",
        "post_dt", "created_utc", "title", "selftext",
        "has_real_body", "num_matched", "multi_ticker_flag",
        "score", "num_comments",
    ]
    links = links[[c for c in keep_cols if c in links.columns]]

    out = interim / "posts_linked_to_events.parquet"
    links.to_parquet(out, index=False)

    per_event = (
        links.groupby(["ticker", "event_id", "rdq"], as_index=False)
             .agg(post_count=("id", "size"),
                  real_body_count=("has_real_body", "sum"))
    )
    all_events = earnings[["ticker", "event_id", "rdq"]].copy()
    per_event_full = all_events.merge(per_event, on=["ticker", "event_id", "rdq"], how="left")
    per_event_full["post_count"] = per_event_full["post_count"].fillna(0).astype(int)
    per_event_full["real_body_count"] = per_event_full["real_body_count"].fillna(0).astype(int)

    per_event_out = interim / "event_post_counts.parquet"
    per_event_full.to_parquet(per_event_out, index=False)

    n_events_total = len(all_events)
    n_events_with_any = int((per_event_full["post_count"] > 0).sum())
    n_events_thin = int((per_event_full["post_count"] < 10).sum())
    n_events_zero = int((per_event_full["post_count"] == 0).sum())

    print(f"earnings events           : {n_events_total:,}")
    print(f"events with >= 1 post     : {n_events_with_any:,}  ({n_events_with_any/n_events_total*100:.1f}%)")
    print(f"events with < 10 posts    : {n_events_thin:,}  ({n_events_thin/n_events_total*100:.1f}%)")
    print(f"events with 0 posts       : {n_events_zero:,}")
    print(f"post-event link rows      : {len(links):,}")
    print(f"distinct posts linked     : {links['id'].nunique():,}")
    print(f"linked with real body     : {int(links['has_real_body'].sum()):,}")
    print()

    stats = (
        per_event_full.groupby("ticker")["post_count"]
        .agg(events="size",
             total="sum",
             mean="mean",
             median="median",
             min="min",
             max="max",
             zero=lambda s: int((s == 0).sum()),
             thin_lt10=lambda s: int((s < 10).sum()))
    )
    body_stats = (
        per_event_full.groupby("ticker")["real_body_count"]
        .agg(rb_total="sum", rb_mean="mean", rb_min="min", rb_max="max")
    )
    stats = stats.join(body_stats).sort_values("total", ascending=False)

    stats_display = stats.copy()
    stats_display["mean"] = stats_display["mean"].round(1)
    stats_display["rb_mean"] = stats_display["rb_mean"].round(1)
    for col in ["median", "min", "max", "zero", "thin_lt10", "rb_min", "rb_max", "events", "total", "rb_total"]:
        stats_display[col] = stats_display[col].astype(int)
    print("per-ticker event-level stats (post_count = matched posts per earnings event):")
    print(stats_display.to_string())
    print()

    stats_out = p["outputs"]["tables"]
    stats_out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_out / "per_ticker_event_stats.csv")

    print(f"wrote: {out}")
    print(f"wrote: {per_event_out}")
    print(f"wrote: {stats_out / 'per_ticker_event_stats.csv'}")


if __name__ == "__main__":
    main()
