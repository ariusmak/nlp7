from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def build_analyzer(wsb_lexicon: dict[str, float] | None = None) -> SentimentIntensityAnalyzer:
    a = SentimentIntensityAnalyzer()
    if wsb_lexicon:
        a.lexicon.update(wsb_lexicon)
    return a


def score_text(analyzer: SentimentIntensityAnalyzer, text: str) -> dict[str, float]:
    if not text:
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(text)


def score_series(analyzer: SentimentIntensityAnalyzer, texts) -> list[dict[str, float]]:
    return [score_text(analyzer, t) for t in texts]


def label_compound(compound: float, pos_thr: float = 0.05, neg_thr: float = -0.05) -> str:
    if compound > pos_thr:
        return "positive"
    if compound < neg_thr:
        return "negative"
    return "neutral"
