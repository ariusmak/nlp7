# outputs/tables

Naming convention: files tagged `_vader` use the WSB-adapted VADER sentiment
scorer. Files tagged `_finbert` use ProsusAI/FinBERT. Files with no tag are
scorer-agnostic (baseline market features, coverage, dataset descriptives).

## scorer-agnostic

- `dataset_summary_totals.csv` — raw post totals and pipeline drop counts.
- `dataset_summary_matched_per_stock.csv` — matched posts per ticker.
- `dataset_summary_window_per_stock.csv` — posts falling inside each ticker's
  10-day pre-earnings event windows.
- `per_ticker_event_stats.csv`, `per_stock_summary.csv` — per-ticker event
  counts and coverage.
- `events_per_ticker_year.csv` — events per ticker per calendar year.
- `eda_sample_posts_per_ticker.csv` — hand-audit sample posts.
- `eda_ambiguous_audit_bare_ticker.csv` — bare-ticker ambiguity audit for
  AMD / BA / DIS / GS / MU.
- `eda_feature_target_correlations.csv` — Pearson r of all numeric features
  with `target_ret3`.
- `market_feature_distributions.csv`, `market_target_per_ticker.csv` —
  baseline market-feature and target descriptives.
- `coefficients_ols_baseline.csv`, `coefficients_ridge_baseline.csv` —
  no-sentiment baseline model coefficients (same across scorers).

## VADER (`_vader`)

- `model_metrics_vader.csv` — 4-variant test/train metrics
  (ols_baseline, ols_sentiment, ridge_baseline, ridge_sentiment)
  computed on the VADER modeling dataset.
- `coefficients_ols_vader.csv`, `coefficients_ridge_vader.csv` —
  sentiment-augmented OLS / Ridge coefficients.
- `ridge_cv_detail_vader.csv` — walk-forward alpha-selection grid.
- `event_sentiment_summary_vader.csv` — mean/median/std of each of the 6
  event-level sentiment features across events with >=1 post.
- `event_sentiment_per_ticker_vader.csv` — per-ticker rollup of coverage and
  mean event-level sentiment.
- `sentiment_audit_top50_movers_vader.csv` — top 50 posts by
  |adapted_compound - raw_compound|.
- `sentiment_label_confusion_vader.csv` — raw-VADER vs WSB-adapted-VADER
  label confusion.
- `sentiment_raw_vs_adapted_vader.csv` — mean / median / std of compound
  before vs after lexicon extension. VADER-only concept (no FinBERT analog).
- `probe_mean_sentiment_only_metrics_vader.csv` — ablation with only
  `mean_sentiment` added.

## FinBERT (`_finbert`)

- `model_metrics_finbert.csv` — same 4-variant schema as the VADER file, with
  sentiment features computed from FinBERT probabilities.
- `coefficients_ols_finbert.csv`, `coefficients_ridge_finbert.csv` —
  sentiment-augmented model coefficients.
- `ridge_cv_detail_finbert.csv` — walk-forward alpha-selection grid.
- `event_sentiment_summary_finbert.csv`,
  `event_sentiment_per_ticker_finbert.csv` — mirror the VADER schema; values
  are computed on `fb_score = p_pos - p_neg`.
- `sentiment_audit_top50_movers_finbert.csv` — top 50 posts by `|fb_score|`,
  with label derived from the 0.05 / -0.05 threshold.
- `sentiment_label_confusion_finbert.csv` — cross-scorer confusion
  (rows = adapted-VADER label, cols = FinBERT label) over all scored posts.
- `probe_finbert_metrics.csv` — the FinBERT feasibility probe run.

## final comparison

- `final_comparison_coefficients_long.csv` — stacked coefficients for
  baseline / VADER-sentiment / FinBERT-sentiment OLS, one row per feature.
- `final_comparison_coefficients_wide.csv` — same data pivoted to one row
  per feature with parallel (coef, pvalue) columns per variant.
- `final_comparison_summary.csv` — R^2, adj R^2, F-stat, train/test RMSE/MAE,
  directional accuracy for the three OLS variants.
