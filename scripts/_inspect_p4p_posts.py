from __future__ import annotations

import sys
from pathlib import Path

PATH = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/p4p_reddit_posts.sql")

COLS = [
    "id", "clicked", "distinguished", "edited", "is_original_content",
    "is_self", "over_18", "selftext", "spoiler", "stickied",
    "title", "upvote_ratio", "created_utc", "num_comments", "score",
    "symbol", "author_id", "sentiment_title_score", "sentiment_body_score",
]

NULL = "NULL"


def parse_tuples(buf: str, start: int, max_rows: int):
    rows = []
    i = start
    n = len(buf)
    while i < n and len(rows) < max_rows:
        if buf[i] != "(":
            i += 1
            continue
        i += 1
        fields = []
        cur = []
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
                    val = "".join(cur).strip()
                    fields.append(("STR" if quoted else "RAW", val))
                    cur, quoted = [], False
                    i += 1
                    continue
                if c == ")":
                    val = "".join(cur).strip()
                    fields.append(("STR" if quoted else "RAW", val))
                    rows.append(fields)
                    i += 1
                    break
                cur.append(c)
                i += 1
        while i < n and buf[i] in ", \n":
            i += 1
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(PATH, "r", encoding="utf-8", errors="replace") as f:
        insert_line = None
        for line in f:
            if line.startswith("INSERT INTO"):
                insert_line = line
                break

    if insert_line is None:
        print("no INSERT found")
        return

    start = insert_line.index("(")
    rows = parse_tuples(insert_line, start, max_rows=5000)
    print(f"parsed {len(rows)} sample tuples, each with {len(rows[0])} fields (expected 19)")
    print()

    selftext_idx = COLS.index("selftext")
    symbol_idx = COLS.index("symbol")
    is_self_idx = COLS.index("is_self")
    title_idx = COLS.index("title")

    n = len(rows)
    n_self = sum(1 for r in rows if r[is_self_idx][1] == "1")
    n_body_null = sum(1 for r in rows if r[selftext_idx][0] == "RAW" and r[selftext_idx][1] == "NULL")
    n_body_empty = sum(1 for r in rows if r[selftext_idx][0] == "STR" and r[selftext_idx][1] == "")
    n_body_text = n - n_body_null - n_body_empty
    n_sym_null = sum(1 for r in rows if r[symbol_idx][0] == "RAW" and r[symbol_idx][1] == "NULL")

    from collections import Counter
    syms = Counter(r[symbol_idx][1] for r in rows if r[symbol_idx][0] == "STR")

    print(f"is_self == 1           : {n_self:,} / {n:,}  ({n_self/n*100:.1f}%)")
    print(f"selftext NULL          : {n_body_null:,} / {n:,}  ({n_body_null/n*100:.1f}%)")
    print(f"selftext empty-string  : {n_body_empty:,} / {n:,}  ({n_body_empty/n*100:.1f}%)")
    print(f"selftext has text      : {n_body_text:,} / {n:,}  ({n_body_text/n*100:.1f}%)")
    print(f"symbol NULL            : {n_sym_null:,} / {n:,}  ({n_sym_null/n*100:.1f}%)")
    print()
    print("top 15 symbol values (when populated):")
    for s, c in syms.most_common(15):
        print(f"  {s!r:>14}  {c:,}")
    print()

    print("=== first 3 rows with body text ===")
    shown = 0
    for r in rows:
        if r[selftext_idx][0] == "STR" and r[selftext_idx][1]:
            title = r[title_idx][1]
            body = r[selftext_idx][1]
            print(f"title : {title[:100]}")
            print(f"body  : {body[:300]}{'...' if len(body) > 300 else ''}")
            print()
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    main()
