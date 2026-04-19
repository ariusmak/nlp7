from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.paths import paths, sentiment_config


CANDIDATES = {
    "SBUX": {"aliases": ["Starbucks"], "ambiguous": False},
    "PFE": {"aliases": ["Pfizer"], "ambiguous": False},
    "COST": {"aliases": ["Costco"], "ambiguous": True},
    "GS": {"aliases": ["Goldman Sachs", "Goldman"], "ambiguous": True},
    "WFC": {"aliases": ["Wells Fargo"], "ambiguous": False},
    "LULU": {"aliases": ["Lululemon"], "ambiguous": False},
    "BB": {"aliases": ["BlackBerry", "Blackberry"], "ambiguous": True},
    "BABA": {"aliases": ["Alibaba"], "ambiguous": False},
    "F": {"aliases": ["Ford"], "ambiguous": True},
    "GM": {"aliases": ["General Motors"], "ambiguous": True},
    "XOM": {"aliases": ["Exxon", "ExxonMobil"], "ambiguous": False},
    "CVX": {"aliases": ["Chevron"], "ambiguous": False},
    "KO": {"aliases": ["Coca-Cola", "Coca Cola"], "ambiguous": True},
    "PG": {"aliases": ["Procter & Gamble", "Procter and Gamble"], "ambiguous": True},
    "T": {"aliases": ["AT&T"], "ambiguous": True},
    "CMG": {"aliases": ["Chipotle"], "ambiguous": False},
}

FINANCE_CUES = ["stock", "shares", "earnings", "calls", "puts", "guidance", "bullish", "bearish", "buy", "sell"]

BL = r"(?<![A-Za-z0-9_])"
BR = r"(?![A-Za-z0-9_])"

cue_re = re.compile(BL + "(?:" + "|".join(FINANCE_CUES) + ")" + BR, re.IGNORECASE)


def build_patterns():
    pats = {}
    for t, info in CANDIDATES.items():
        cash = re.compile(BL + r"\$" + t + BR, re.IGNORECASE)
        aliases = info["aliases"] + [t]
        alias = re.compile(BL + "(?:" + "|".join(re.escape(a) for a in aliases) + ")" + BR, re.IGNORECASE)
        bare = re.compile(BL + t + BR, re.IGNORECASE)
        pats[t] = (cash, alias, bare, info["ambiguous"])
    return pats


def match_row(text: str, pats) -> list[str]:
    if not text:
        return []
    has_cue = bool(cue_re.search(text))
    hits = []
    for t, (cash, alias, bare, ambig) in pats.items():
        if cash.search(text) or alias.search(text):
            hits.append(t); continue
        if bare.search(text):
            if ambig:
                if has_cue:
                    hits.append(t)
            else:
                hits.append(t)
    return hits


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    posts = pd.read_parquet(p["data"]["interim"] / "reddit_posts_in_windows.parquet")
    posts["title"] = posts["title"].fillna("").astype(str)
    posts["selftext"] = posts["selftext"].fillna("").astype(str)
    text = (posts["title"] + " " + posts["selftext"]).astype(str)

    pats = build_patterns()
    from collections import Counter
    per_ticker = Counter()
    per_ticker_with_body = Counter()
    body_stripped = posts["selftext"].str.strip()
    PLACEHOLDERS = {"[removed]", "[deleted]"}
    has_real_body = (~body_stripped.isin(PLACEHOLDERS)) & (body_stripped.str.len() >= 20)

    for i, t in enumerate(text.values):
        hits = match_row(t, pats)
        rb = bool(has_real_body.iat[i])
        for h in hits:
            per_ticker[h] += 1
            if rb:
                per_ticker_with_body[h] += 1
        if i % 100000 == 0 and i > 0:
            sys.stderr.write(f"  ...{i:,}/{len(text):,}\n")

    print(f"scanned {len(text):,} in-window posts")
    print()
    print(f"{'ticker':>6}  {'matches':>10}  {'real_body':>10}  ambiguous")
    for t in sorted(per_ticker, key=per_ticker.get, reverse=True):
        print(f"{t:>6}  {per_ticker[t]:>10,}  {per_ticker_with_body[t]:>10,}  {CANDIDATES[t]['ambiguous']}")
    print()
    print("candidates missing:")
    for t in CANDIDATES:
        if t not in per_ticker:
            print(f"  {t}: 0 matches")


if __name__ == "__main__":
    main()
