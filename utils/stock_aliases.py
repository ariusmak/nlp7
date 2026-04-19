from __future__ import annotations

from utils.paths import universe


def _entries() -> list[dict]:
    return universe()["tickers"]


def tickers() -> list[str]:
    return [e["ticker"] for e in _entries()]


def ambiguous_set() -> set[str]:
    return {e["ticker"] for e in _entries() if e.get("ambiguous", False)}


def name_aliases_by_ticker() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in _entries():
        t = e["ticker"]
        aliases = [a for a in e.get("aliases", []) if a.upper() != t.upper()]
        out[t] = aliases
    return out


def finance_cues() -> list[str]:
    return list(universe().get("finance_cues", []))
