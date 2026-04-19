from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from utils.stock_aliases import (
    tickers,
    ambiguous_set,
    name_aliases_by_ticker,
    finance_cues,
)


_BOUNDARY_LEFT = r"(?<![A-Za-z0-9_])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9_])"


def _alias_pattern(aliases: Iterable[str]) -> re.Pattern | None:
    escaped = [re.escape(a) for a in aliases]
    if not escaped:
        return None
    body = "|".join(escaped)
    return re.compile(_BOUNDARY_LEFT + f"(?:{body})" + _BOUNDARY_RIGHT, re.IGNORECASE)


def _cashtag_pattern(ticker: str) -> re.Pattern:
    return re.compile(_BOUNDARY_LEFT + r"\$" + ticker + _BOUNDARY_RIGHT, re.IGNORECASE)


def _ticker_pattern(ticker: str) -> re.Pattern:
    return re.compile(_BOUNDARY_LEFT + ticker + _BOUNDARY_RIGHT, re.IGNORECASE)


def _cue_pattern(cues: Iterable[str]) -> re.Pattern:
    body = "|".join(re.escape(c) for c in cues)
    return re.compile(_BOUNDARY_LEFT + f"(?:{body})" + _BOUNDARY_RIGHT, re.IGNORECASE)


@dataclass
class StockMatcher:
    tickers: list[str]
    ambiguous: set[str]
    cashtag_re: dict[str, re.Pattern]
    alias_re: dict[str, re.Pattern | None]
    ticker_re: dict[str, re.Pattern]
    cue_re: re.Pattern

    @classmethod
    def build(cls) -> "StockMatcher":
        ts = tickers()
        name_aliases = name_aliases_by_ticker()
        return cls(
            tickers=ts,
            ambiguous=ambiguous_set(),
            cashtag_re={t: _cashtag_pattern(t) for t in ts},
            alias_re={t: _alias_pattern(name_aliases[t]) for t in ts},
            ticker_re={t: _ticker_pattern(t) for t in ts},
            cue_re=_cue_pattern(finance_cues()),
        )

    def match(self, text: str) -> list[str]:
        if not text:
            return []
        has_cue = bool(self.cue_re.search(text))
        hits: list[str] = []
        for t in self.tickers:
            cash = self.cashtag_re[t].search(text) is not None
            alias_re = self.alias_re[t]
            alias = alias_re.search(text) is not None if alias_re else False
            bare = self.ticker_re[t].search(text) is not None
            if cash or alias:
                hits.append(t)
                continue
            if bare:
                if t in self.ambiguous:
                    if has_cue:
                        hits.append(t)
                else:
                    hits.append(t)
        return hits

    def match_with_sources(self, text: str) -> dict[str, list[str]]:
        if not text:
            return {}
        has_cue = bool(self.cue_re.search(text))
        out: dict[str, list[str]] = {}
        for t in self.tickers:
            sources: list[str] = []
            if self.cashtag_re[t].search(text):
                sources.append("cashtag")
            alias_re = self.alias_re[t]
            if alias_re and alias_re.search(text):
                sources.append("alias")
            if self.ticker_re[t].search(text):
                if t in self.ambiguous:
                    if has_cue and not ("cashtag" in sources or "alias" in sources):
                        sources.append("ticker_plus_cue")
                    elif not ("cashtag" in sources or "alias" in sources):
                        pass
                else:
                    if not ("cashtag" in sources or "alias" in sources):
                        sources.append("ticker")
            if sources:
                out[t] = sources
        return out
