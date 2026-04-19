from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

PATH = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/p4p_reddit_posts.sql")
SELFTEXT_IDX = 7


def classify(kind: str, val: str) -> str:
    if kind == "RAW" and val == "NULL":
        return "null"
    if kind == "STR":
        if val == "":
            return "empty"
        if val == "[removed]":
            return "removed_placeholder"
        if val == "[deleted]":
            return "deleted_placeholder"
        n = len(val)
        if n < 20:
            return "very_short"
        if n < 200:
            return "short"
        if n < 1000:
            return "medium"
        return "long"
    return "other"


def iter_selftext(line: str):
    i = line.index("(")
    n = len(line)
    while i < n:
        if line[i] != "(":
            i += 1
            continue
        i += 1
        field_idx = 0
        cur: list[str] = []
        in_str = False
        quoted = False
        while i < n:
            c = line[i]
            if in_str:
                if c == "\\" and i + 1 < n:
                    cur.append(line[i + 1])
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
                    if field_idx == SELFTEXT_IDX:
                        val = "".join(cur).strip()
                        yield ("STR" if quoted else "RAW", val)
                        cur = []
                        quoted = False
                        while i < n and line[i] != ")":
                            if line[i] == "'" and not in_str:
                                in_str = True
                                i += 1
                                continue
                            if in_str:
                                if line[i] == "\\" and i + 1 < n:
                                    i += 2; continue
                                if line[i] == "'":
                                    in_str = False
                                    i += 1; continue
                                i += 1; continue
                            i += 1
                        i += 1
                        while i < n and line[i] in ", \n":
                            i += 1
                        break
                    cur = []
                    quoted = False
                    field_idx += 1
                    i += 1
                    continue
                if c == ")":
                    i += 1
                    while i < n and line[i] in ", \n":
                        i += 1
                    break
                cur.append(c)
                i += 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    counter: Counter[str] = Counter()
    total = 0
    total_len_chars = 0
    total_content_rows = 0
    with open(PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            for kind, val in iter_selftext(line):
                c = classify(kind, val)
                counter[c] += 1
                total += 1
                if c in {"short", "medium", "long", "very_short"}:
                    total_len_chars += len(val)
                    total_content_rows += 1
            if total % 200000 < 10000:
                sys.stderr.write(f"...{total:,} rows so far\n")

    print(f"rows scanned : {total:,}")
    print()
    print("selftext classification:")
    order = ["null", "empty", "removed_placeholder", "deleted_placeholder",
             "very_short", "short", "medium", "long", "other"]
    for k in order:
        v = counter.get(k, 0)
        print(f"  {k:>22}  {v:>10,}  ({v/total*100:5.1f}%)")
    real = counter.get("short", 0) + counter.get("medium", 0) + counter.get("long", 0)
    print()
    print(f"rows with real body text (>= 20 chars, not placeholder): {real:,}  ({real/total*100:.1f}%)")
    if total_content_rows:
        print(f"mean body length among content rows : {total_len_chars/total_content_rows:.0f} chars")


if __name__ == "__main__":
    main()
