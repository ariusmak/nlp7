from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.stock_matcher import StockMatcher

PATH = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/p4p_reddit_posts.sql")

SELFTEXT_IDX = 7
TITLE_IDX = 10
CREATED_IDX = 12

PLACEHOLDERS = {"[removed]", "[deleted]"}


def parse_row(buf: str, i: int):
    n = len(buf)
    if buf[i] != "(":
        return None, i
    i += 1
    field_idx = 0
    fields: list[tuple[str, str]] = []
    cur: list[str] = []
    in_str = False
    quoted = False
    while i < n:
        c = buf[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                cur.append(buf[i + 1])
                i += 2
                continue
            if c == "'":
                in_str = False
                i += 1
                continue
            cur.append(c)
            i += 1
        else:
            if c == "'":
                in_str = True
                quoted = True
                i += 1
                continue
            if c == ",":
                fields.append(("STR" if quoted else "RAW", "".join(cur).strip()))
                cur = []
                quoted = False
                field_idx += 1
                i += 1
                continue
            if c == ")":
                fields.append(("STR" if quoted else "RAW", "".join(cur).strip()))
                i += 1
                while i < n and buf[i] in ", \n":
                    i += 1
                return fields, i
            cur.append(c)
            i += 1
    return None, i


def has_real_body(kind: str, val: str) -> bool:
    if kind == "RAW":
        return False
    if val in PLACEHOLDERS or not val:
        return False
    return len(val) >= 20


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    matcher = StockMatcher.build()

    from utils.paths import universe
    safe = universe()["safe_window"]
    win_start = str(safe["start"])
    win_end_cap = "2023-03-28"

    total = 0
    with_body = 0
    matched_any = 0
    per_ticker = Counter()
    per_year = Counter()
    matched_with_body = 0
    filtered_in_window = 0
    last_report = time.time()

    with open(PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            try:
                i = line.index("(")
            except ValueError:
                continue
            while i < len(line):
                fields, i = parse_row(line, i)
                if fields is None or len(fields) < 19:
                    break
                total += 1

                title_kind, title_val = fields[TITLE_IDX]
                body_kind, body_val = fields[SELFTEXT_IDX]
                created_kind, created_val = fields[CREATED_IDX]

                title = title_val if title_kind == "STR" else ""
                body = body_val if body_kind == "STR" and body_val not in PLACEHOLDERS else ""
                created = created_val if created_kind == "STR" else ""

                if not created or created < win_start or created > win_end_cap + " 23:59:59":
                    continue
                filtered_in_window += 1

                if has_real_body(body_kind, body_val):
                    with_body += 1

                text = title + " " + body
                hits = matcher.match(text)
                if hits:
                    matched_any += 1
                    year = int(created[:4])
                    per_year[year] += 1
                    has_body_here = has_real_body(body_kind, body_val)
                    if has_body_here:
                        matched_with_body += 1
                    for t in hits:
                        per_ticker[t] += 1

                if time.time() - last_report > 5:
                    last_report = time.time()
                    sys.stderr.write(f"  ...{total:,} parsed, {filtered_in_window:,} in-window, {matched_any:,} matched\n")

    print(f"rows parsed                 : {total:,}")
    print(f"rows in date window         : {filtered_in_window:,}  ({filtered_in_window/total*100:.1f}%)")
    print(f"rows with real body         : {with_body:,}  ({with_body/filtered_in_window*100:.1f}% of in-window)")
    print(f"rows matched to any ticker  : {matched_any:,}  ({matched_any/filtered_in_window*100:.1f}% of in-window)")
    print(f"matched AND body            : {matched_with_body:,}")
    print()
    print("per-ticker matches:")
    for t in sorted(per_ticker, key=per_ticker.get, reverse=True):
        print(f"  {t:>6}  {per_ticker[t]:>8,}")
    print()
    print("matched posts per year:")
    for y in sorted(per_year):
        print(f"  {y}  {per_year[y]:>8,}")


if __name__ == "__main__":
    main()
