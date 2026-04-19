from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


POSTS_CSV = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/posts.csv")
WSB2022_CSV = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/wallstreetbets_2022.csv")

TICKERS = ["AAPL", "GME", "MCD", "MSFT", "NFLX", "NVDA", "TSLA"]
ALIASES = {
    "AAPL": ["Apple"],
    "GME": ["GameStop", "Game Stop"],
    "MCD": ["McDonald's", "McDonalds"],
    "MSFT": ["Microsoft"],
    "NFLX": ["Netflix"],
    "NVDA": ["Nvidia", "NVIDIA"],
    "TSLA": ["Tesla"],
}
BOUNDARY_L = r"(?<![A-Za-z0-9_])"
BOUNDARY_R = r"(?![A-Za-z0-9_])"

CASHTAG_RE = {t: re.compile(BOUNDARY_L + r"\$" + t + BOUNDARY_R, re.IGNORECASE) for t in TICKERS}
TICKER_RE = {t: re.compile(BOUNDARY_L + t + BOUNDARY_R, re.IGNORECASE) for t in TICKERS}
ALIAS_RE = {
    t: re.compile(BOUNDARY_L + "(?:" + "|".join(re.escape(a) for a in ALIASES[t]) + ")" + BOUNDARY_R, re.IGNORECASE)
    for t in TICKERS
}

PLACEHOLDERS = {"[removed]", "[deleted]"}


def has_real_body(body: str) -> bool:
    if not body:
        return False
    s = body.strip()
    if s in PLACEHOLDERS:
        return False
    return len(s) >= 20


def match_tickers(text: str) -> list[str]:
    if not text:
        return []
    hits = []
    for t in TICKERS:
        if CASHTAG_RE[t].search(text) or ALIAS_RE[t].search(text) or TICKER_RE[t].search(text):
            hits.append(t)
    return hits


def load_posts_csv() -> pd.DataFrame:
    df = pd.read_csv(POSTS_CSV, low_memory=False)
    df = df[df["subreddit"].fillna("").str.lower() == "wallstreetbets"].copy()
    df = df.rename(columns={"selftext": "body"})
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df["source_file"] = "posts.csv"
    return df[["id", "created_dt", "subreddit", "title", "body", "source_file"]]


def load_wsb2022_csv() -> pd.DataFrame:
    df = pd.read_csv(WSB2022_CSV, low_memory=False)
    df = df[df["title"].fillna("") != "Comment"].copy()
    df["created_dt"] = pd.to_datetime(df["created"], unit="s", utc=True, errors="coerce")
    df = df[df["created_dt"] >= pd.Timestamp("2023-01-01", tz="UTC")].copy()
    df["subreddit"] = "wallstreetbets"
    df["source_file"] = "wallstreetbets_2022.csv"
    return df[["id", "created_dt", "subreddit", "title", "body", "source_file"]]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    a = load_posts_csv()
    b = load_wsb2022_csv()
    combined = pd.concat([a, b], ignore_index=True)
    combined["title"] = combined["title"].fillna("").astype(str)
    combined["body"] = combined["body"].fillna("").astype(str)
    combined = combined.drop_duplicates(subset=["id"]).reset_index(drop=True)
    combined["text"] = combined["title"] + " " + combined["body"]

    combined["has_body"] = combined["body"].map(has_real_body)
    combined["matched"] = combined["text"].map(match_tickers)
    combined["num_matched"] = combined["matched"].map(len)

    total = len(combined)
    n_a = len(a); n_b = len(b)
    n_body = int(combined["has_body"].sum())
    n_any = int((combined["num_matched"] > 0).sum())

    print(f"posts.csv (2018-2022)          : {n_a:>10,}")
    print(f"wsb_2022 posts-only (>=2023)   : {n_b:>10,}")
    print(f"combined (after id dedup)      : {total:>10,}")
    print()
    print(f"posts with real body text      : {n_body:>10,}  ({n_body/total*100:5.1f}%)")
    print(f"posts matched to 1+ of 7 tickers: {n_any:>10,}  ({n_any/total*100:5.1f}%)")
    print()
    print("per-ticker post counts (counts multi-ticker posts in each):")
    from collections import Counter
    per_ticker = Counter()
    for lst in combined["matched"]:
        for t in lst:
            per_ticker[t] += 1
    for t in TICKERS:
        print(f"  {t:>6}  {per_ticker[t]:>8,}")
    print()

    matched = combined[combined["num_matched"] > 0]
    print(f"within matched subset ({len(matched):,} posts):")
    print(f"  real body text          : {int(matched['has_body'].sum()):,}  ({matched['has_body'].mean()*100:.1f}%)")
    print(f"  empty body              : {int((matched['body'].str.strip() == '').sum()):,}")
    print(f"  [removed]/[deleted]     : {int(matched['body'].str.strip().isin(PLACEHOLDERS).sum()):,}")
    print()

    yearly = matched.groupby(matched["created_dt"].dt.year).size()
    print("matched posts per year:")
    for year, n in yearly.items():
        print(f"  {int(year)}  {n:>8,}")
    print()

    print("subreddit distribution in matched subset:")
    print(matched["subreddit"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
