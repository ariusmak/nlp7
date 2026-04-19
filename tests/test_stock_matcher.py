from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.stock_matcher import StockMatcher


def _m():
    return StockMatcher.build()


def test_cashtag_simple():
    assert _m().match("$AAPL looking strong") == ["AAPL"]


def test_bare_ticker_nonambiguous():
    assert "TSLA" in _m().match("TSLA to the moon")


def test_alias_match():
    assert "GOOGL" in _m().match("Google will announce earnings tomorrow")


def test_ambiguous_amd_no_cue_dropped():
    assert _m().match("AMD") == []


def test_ambiguous_amd_with_cue_kept():
    assert "AMD" in _m().match("AMD earnings beat")


def test_ambiguous_amd_cashtag_accepted():
    assert "AMD" in _m().match("$AMD yolo")


def test_ambiguous_amd_alias_accepted():
    assert "AMD" in _m().match("Advanced Micro Devices posts strong Q3")


def test_ambiguous_dis_without_cue_dropped():
    assert _m().match("DIS to the moon") == []


def test_ambiguous_dis_alias_accepted():
    assert "DIS" in _m().match("Disney parks reopening")


def test_lowercase_bare_matched():
    assert "TSLA" in _m().match("tsla is trending")


def test_mixed_case_bare_matched():
    assert "AAPL" in _m().match("Aapl just reported")


def test_multi_ticker():
    hits = _m().match("AAPL MSFT and TSLA calls ripping")
    assert set(hits) >= {"AAPL", "MSFT", "TSLA"}


def test_plural_not_matched():
    assert "AAPL" not in _m().match("AAPLs are apples")


def test_empty_and_none():
    m = _m()
    assert m.match("") == []
    assert m.match(None) == []
