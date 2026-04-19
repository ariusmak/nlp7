from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.paths import paths, sentiment_config
from utils.sentiment_scoring import build_analyzer, label_compound
from utils.wsb_preprocessing import (
    build_phrase_regexes,
    clean_text,
    combine_title_body,
    apply_emoji_map,
    apply_phrase_regexes,
    load_emoji_map,
    load_phrase_map,
    load_wsb_lexicon,
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    cfg = sentiment_config()

    pos_thr = float(cfg["compound_thresholds"]["positive"])
    neg_thr = float(cfg["compound_thresholds"]["negative"])
    use_phrase = bool(cfg.get("use_phrase_map", True))
    use_emoji = bool(cfg.get("use_emoji_map", True))
    use_lexicon = bool(cfg.get("use_custom_lexicon", True))
    emit_audit = bool(cfg.get("emit_audit_table", True))

    interim = p["data"]["interim"]
    processed = p["data"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    posts = pd.read_parquet(interim / "posts_linked_to_events.parquet")
    print(f"loaded {len(posts):,} post-event link rows")

    phrase_map = load_phrase_map() if use_phrase else {}
    emoji_map = load_emoji_map() if use_emoji else {}
    wsb_lex = load_wsb_lexicon() if use_lexicon else {}
    phrase_regexes = build_phrase_regexes(phrase_map)

    raw_analyzer = build_analyzer(wsb_lexicon=None)
    adapted_analyzer = build_analyzer(wsb_lexicon=wsb_lex)
    print(f"vader lexicon: raw={len(raw_analyzer.lexicon):,}, adapted={len(adapted_analyzer.lexicon):,} (+{len(adapted_analyzer.lexicon) - len(raw_analyzer.lexicon)})")

    titles = posts["title"].fillna("").astype(str).tolist()
    bodies = posts["selftext"].fillna("").astype(str).tolist()

    raw_texts: list[str] = []
    proc_texts: list[str] = []
    for t, b in tqdm(zip(titles, bodies), total=len(titles), desc="preprocess"):
        combined = combine_title_body(t, b)
        cleaned = clean_text(combined)
        raw_texts.append(cleaned)
        with_emojis = apply_emoji_map(cleaned, emoji_map)
        with_phrases = apply_phrase_regexes(with_emojis, phrase_regexes)
        proc_texts.append(with_phrases)

    raw_scores = np.empty((len(raw_texts), 4), dtype=np.float32)
    adp_scores = np.empty((len(proc_texts), 4), dtype=np.float32)
    for i, t in enumerate(tqdm(raw_texts, desc="vader raw")):
        s = raw_analyzer.polarity_scores(t) if t else {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        raw_scores[i] = (s["neg"], s["neu"], s["pos"], s["compound"])
    for i, t in enumerate(tqdm(proc_texts, desc="vader adapted")):
        s = adapted_analyzer.polarity_scores(t) if t else {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        adp_scores[i] = (s["neg"], s["neu"], s["pos"], s["compound"])

    scored = posts.copy()
    scored["text_clean"] = raw_texts
    scored["text_processed"] = proc_texts
    scored["raw_neg"] = raw_scores[:, 0]
    scored["raw_neu"] = raw_scores[:, 1]
    scored["raw_pos"] = raw_scores[:, 2]
    scored["raw_compound"] = raw_scores[:, 3]
    scored["adp_neg"] = adp_scores[:, 0]
    scored["adp_neu"] = adp_scores[:, 1]
    scored["adp_pos"] = adp_scores[:, 2]
    scored["adp_compound"] = adp_scores[:, 3]
    scored["raw_label"] = [label_compound(c, pos_thr, neg_thr) for c in scored["raw_compound"]]
    scored["adp_label"] = [label_compound(c, pos_thr, neg_thr) for c in scored["adp_compound"]]
    scored["compound_delta"] = scored["adp_compound"] - scored["raw_compound"]

    out = processed / "posts_scored.parquet"
    scored.to_parquet(out, index=False)
    print(f"wrote: {out}  ({len(scored):,} rows)")

    tables_dir = p["outputs"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    nonempty = scored[scored["text_clean"].str.len() > 0]
    summary = pd.DataFrame({
        "metric": ["mean_compound", "median_compound", "std_compound",
                   "frac_positive", "frac_neutral", "frac_negative"],
        "raw": [
            nonempty["raw_compound"].mean(),
            nonempty["raw_compound"].median(),
            nonempty["raw_compound"].std(),
            (nonempty["raw_label"] == "positive").mean(),
            (nonempty["raw_label"] == "neutral").mean(),
            (nonempty["raw_label"] == "negative").mean(),
        ],
        "adapted": [
            nonempty["adp_compound"].mean(),
            nonempty["adp_compound"].median(),
            nonempty["adp_compound"].std(),
            (nonempty["adp_label"] == "positive").mean(),
            (nonempty["adp_label"] == "neutral").mean(),
            (nonempty["adp_label"] == "negative").mean(),
        ],
    })
    summary["delta"] = summary["adapted"] - summary["raw"]
    print()
    print("sentiment summary (non-empty cleaned posts, N={:,}):".format(len(nonempty)))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(tables_dir / "sentiment_raw_vs_adapted.csv", index=False)

    print()
    print("confusion of labels (rows=raw, cols=adapted):")
    confusion = pd.crosstab(nonempty["raw_label"], nonempty["adp_label"], dropna=False)
    print(confusion.to_string())
    confusion.to_csv(tables_dir / "sentiment_label_confusion.csv")

    if emit_audit:
        movers = nonempty.reindex(
            nonempty["compound_delta"].abs().sort_values(ascending=False).index
        ).head(50)
        audit_cols = ["ticker", "event_id", "rdq", "post_dt", "score", "num_comments",
                      "raw_compound", "adp_compound", "compound_delta",
                      "raw_label", "adp_label", "text_clean", "text_processed"]
        audit_cols = [c for c in audit_cols if c in movers.columns]
        audit = movers[audit_cols].copy()
        for c in ("text_clean", "text_processed"):
            if c in audit.columns:
                audit[c] = audit[c].str.slice(0, 400)
        audit.to_csv(tables_dir / "sentiment_audit_top50_movers.csv", index=False)
        print()
        print(f"wrote audit: {tables_dir / 'sentiment_audit_top50_movers.csv'}")

    print()
    print(f"wrote: {tables_dir / 'sentiment_raw_vs_adapted.csv'}")
    print(f"wrote: {tables_dir / 'sentiment_label_confusion.csv'}")


if __name__ == "__main__":
    main()
