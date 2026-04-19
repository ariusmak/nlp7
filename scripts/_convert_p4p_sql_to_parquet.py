from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/p4p_reddit_posts.sql")
OUT = Path(r"c:/Users/arius/Desktop/NLP/data/raw/reddit/p4p_reddit_posts.parquet")

ID_IDX = 0
SELFTEXT_IDX = 7
TITLE_IDX = 10
CREATED_IDX = 12
SCORE_IDX = 14
NUM_COMMENTS_IDX = 13
SYMBOL_IDX = 15

EXPECTED_FIELDS = 19
BATCH_ROWS = 50_000


def parse_row(buf: str, i: int):
    n = len(buf)
    if buf[i] != "(":
        return None, i
    i += 1
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


def str_or_none(kind: str, val: str) -> str | None:
    if kind == "RAW" and val == "NULL":
        return None
    return val


def int_or_none(kind: str, val: str) -> int | None:
    if kind == "RAW" and val == "NULL":
        return None
    try:
        return int(val)
    except ValueError:
        return None


SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("selftext", pa.string()),
    pa.field("created_utc", pa.timestamp("s")),
    pa.field("score", pa.int64()),
    pa.field("num_comments", pa.int64()),
    pa.field("symbol", pa.string()),
])


def flush(writer, batch: dict[str, list]) -> int:
    if not batch["id"]:
        return 0
    ts = pd.to_datetime(pd.Series(batch["created_utc"]), errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    table = pa.table({
        "id": batch["id"],
        "title": batch["title"],
        "selftext": batch["selftext"],
        "created_utc": ts.to_list(),
        "score": batch["score"],
        "num_comments": batch["num_comments"],
        "symbol": batch["symbol"],
    }, schema=SCHEMA)
    writer.write_table(table)
    n = len(batch["id"])
    for k in batch:
        batch[k].clear()
    return n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    batch = {k: [] for k in ["id", "title", "selftext", "created_utc", "score", "num_comments", "symbol"]}

    total_parsed = 0
    total_written = 0
    last_report = time.time()

    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd")
    try:
        with open(SRC, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("INSERT INTO"):
                    continue
                try:
                    i = line.index("(")
                except ValueError:
                    continue
                while i < len(line):
                    fields, i = parse_row(line, i)
                    if fields is None or len(fields) < EXPECTED_FIELDS:
                        break
                    total_parsed += 1

                    id_val = str_or_none(*fields[ID_IDX])
                    if id_val is None:
                        continue
                    title_val = str_or_none(*fields[TITLE_IDX]) or ""
                    sel_val = str_or_none(*fields[SELFTEXT_IDX]) or ""
                    created_val = str_or_none(*fields[CREATED_IDX])
                    score_val = int_or_none(*fields[SCORE_IDX])
                    nc_val = int_or_none(*fields[NUM_COMMENTS_IDX])
                    sym_val = str_or_none(*fields[SYMBOL_IDX])

                    batch["id"].append(id_val)
                    batch["title"].append(title_val)
                    batch["selftext"].append(sel_val)
                    batch["created_utc"].append(created_val)
                    batch["score"].append(score_val)
                    batch["num_comments"].append(nc_val)
                    batch["symbol"].append(sym_val)

                    if len(batch["id"]) >= BATCH_ROWS:
                        total_written += flush(writer, batch)

                    if time.time() - last_report > 5:
                        last_report = time.time()
                        sys.stderr.write(f"  ...{total_parsed:,} parsed, {total_written + len(batch['id']):,} queued\n")
        total_written += flush(writer, batch)
    finally:
        writer.close()

    print(f"rows parsed  : {total_parsed:,}")
    print(f"rows written : {total_written:,}")
    print(f"wrote        : {OUT}")
    print(f"size         : {OUT.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
