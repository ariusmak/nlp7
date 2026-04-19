from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from utils.paths import CONFIGS_DIR


PLACEHOLDERS = {"[removed]", "[deleted]", "removed", "deleted"}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_WS_RE = re.compile(r"\s+")
_BOUNDARY_L = r"(?<![A-Za-z0-9_])"
_BOUNDARY_R = r"(?![A-Za-z0-9_])"


def load_phrase_map(path: Path | None = None) -> dict[str, str]:
    path = path or (CONFIGS_DIR / "wsb_phrase_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_emoji_map(path: Path | None = None) -> dict[str, str]:
    path = path or (CONFIGS_DIR / "wsb_emoji_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wsb_lexicon(path: Path | None = None) -> dict[str, float]:
    path = path or (CONFIGS_DIR / "wsb_vader_lexicon.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def is_placeholder_body(s: str) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    return t == "" or t.lower() in PLACEHOLDERS


def clean_text(raw: str) -> str:
    if raw is None:
        return ""
    t = str(raw)
    t = unicodedata.normalize("NFKC", t)
    t = _MD_IMG_RE.sub(" ", t)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _URL_RE.sub(" ", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = _WS_RE.sub(" ", t).strip()
    return t


def _compile_phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(_BOUNDARY_L + re.escape(phrase) + _BOUNDARY_R, re.IGNORECASE)


def build_phrase_regexes(phrase_map: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    items = sorted(phrase_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    return [(_compile_phrase_pattern(k), v) for k, v in items]


def apply_emoji_map(text: str, emoji_map: dict[str, str]) -> str:
    if not text:
        return ""
    items = sorted(emoji_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    for emoji, token in items:
        if emoji in text:
            text = text.replace(emoji, " " + token + " ")
    return _WS_RE.sub(" ", text).strip()


def apply_phrase_regexes(text: str, regexes: list[tuple[re.Pattern, str]]) -> str:
    if not text:
        return ""
    for pat, token in regexes:
        text = pat.sub(token, text)
    return text


def combine_title_body(title: str, body: str) -> str:
    title = "" if title is None else str(title)
    body = "" if body is None or is_placeholder_body(body) else str(body)
    return (title + " " + body).strip()


def preprocess_for_sentiment(
    title: str,
    body: str,
    emoji_map: dict[str, str],
    phrase_regexes: list[tuple[re.Pattern, str]],
) -> tuple[str, str]:
    combined = combine_title_body(title, body)
    cleaned = clean_text(combined)
    with_emojis = apply_emoji_map(cleaned, emoji_map)
    with_phrases = apply_phrase_regexes(with_emojis, phrase_regexes)
    return cleaned, with_phrases
