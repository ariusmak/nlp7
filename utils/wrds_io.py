from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils.paths import paths, universe


COMPUSTAT_QUERY = """
SELECT tic AS ticker,
       datadate,
       fyearq,
       fqtr,
       rdq
FROM comp.fundq
WHERE tic IN ({ticker_list})
  AND rdq IS NOT NULL
  AND rdq BETWEEN '{start}' AND '{end}'
  AND indfmt = 'INDL'
  AND datafmt = 'STD'
  AND popsrc = 'D'
  AND consol = 'C'
ORDER BY tic, rdq
"""


def _universe_tickers() -> list[str]:
    return [row["ticker"] for row in universe()["tickers"]]


def _safe_window() -> tuple[str, str]:
    w = universe()["safe_window"]
    return str(w["start"]), str(w["end"])


def load_earnings_dates_from_file() -> pd.DataFrame | None:
    p = paths()["files"]
    for candidate in (p["earnings_dates_parquet"], p["earnings_dates_csv"]):
        if Path(candidate).exists():
            if str(candidate).endswith(".parquet"):
                df = pd.read_parquet(candidate)
            else:
                df = pd.read_csv(candidate)
            return _normalize(df)
    return None


def _wrds_username_from_pgpass() -> str | None:
    import os
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "postgresql" / "pgpass.conf",
        Path.home() / ".pgpass",
    ]
    for cand in candidates:
        if cand.exists():
            for line in cand.read_text().splitlines():
                parts = line.strip().split(":")
                if len(parts) >= 5 and "wharton.upenn.edu" in parts[0]:
                    return parts[3]
    return None


def pull_earnings_dates_from_wrds() -> pd.DataFrame:
    import wrds

    tickers = _universe_tickers()
    start, end = _safe_window()

    username = _wrds_username_from_pgpass()
    conn = wrds.Connection(wrds_username=username) if username else wrds.Connection()
    try:
        ticker_list = ", ".join(f"'{t}'" for t in tickers)
        sql = COMPUSTAT_QUERY.format(ticker_list=ticker_list, start=start, end=end)
        df = conn.raw_sql(sql)
    finally:
        conn.close()
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "rdq" in df.columns:
        df["rdq"] = pd.to_datetime(df["rdq"]).dt.tz_localize(None).dt.normalize()
    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["rdq"]).reset_index(drop=True)
    df["earnings_timing"] = "unknown"
    df["timing_source"] = "none"
    return df[["ticker", "datadate", "fyearq", "fqtr", "rdq",
               "earnings_timing", "timing_source"]]


def load_earnings_dates() -> tuple[pd.DataFrame, str]:
    df = load_earnings_dates_from_file()
    if df is not None:
        return df, "file"
    df = pull_earnings_dates_from_wrds()
    return df, "wrds"


CRSP_DAILY_QUERY = """
SELECT d.permno,
       n.ticker,
       d.date,
       d.prc,
       d.ret,
       d.vol,
       d.cfacpr,
       d.cfacshr
FROM crsp.dsf d
JOIN crsp.dsenames n ON d.permno = n.permno
WHERE n.ticker IN ({ticker_list})
  AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, DATE '9999-12-31')
  AND d.date BETWEEN '{start}' AND '{end}'
ORDER BY n.ticker, d.date
"""


def pull_crsp_daily_from_wrds(start: str, end: str, tickers: list[str] | None = None) -> pd.DataFrame:
    import wrds

    tickers = tickers or _universe_tickers()

    username = _wrds_username_from_pgpass()
    conn = wrds.Connection(wrds_username=username) if username else wrds.Connection()
    try:
        ticker_list = ", ".join(f"'{t}'" for t in tickers)
        sql = CRSP_DAILY_QUERY.format(ticker_list=ticker_list, start=start, end=end)
        df = conn.raw_sql(sql)
    finally:
        conn.close()

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["prc"] = df["prc"].abs()
    for c in ("ret", "vol", "cfacpr", "cfacshr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
