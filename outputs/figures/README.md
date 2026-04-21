# outputs/figures

Naming convention follows `outputs/tables/`: `_vader` = WSB-adapted VADER,
`_finbert` = ProsusAI/FinBERT. Unsuffixed files are scorer-agnostic.

## scorer-agnostic

- `01_coverage_per_ticker.png` — share of events with >=1 post, per ticker.
- `02_post_count_box_per_ticker.png` — boxplot of event `post_count` by
  ticker.
- `05_target_distribution.png` — histogram of `target_ret3`.

## VADER (`_vader`)

- `03_sentiment_hist_raw_vs_adapted_vader.png` — post-level VADER compound
  distribution before vs after the WSB lexicon extension. VADER-only concept.
- `04_event_sentiment_vs_target_vader.png` — scatter of event-level
  `mean_sentiment` (adapted VADER) vs `target_ret3`, restricted to events
  with >=10 posts.
- `06_predictions_vs_actual_vader.png` — 2x2 grid of predicted vs actual
  returns for OLS/Ridge x baseline/sentiment on the 2022-2023 test set.
- `07_residuals_by_variant_vader.png` — boxplot of test-set residuals per
  variant.

## FinBERT (`_finbert`)

- `04_event_sentiment_vs_target_finbert.png` — same scatter as the VADER
  figure but `mean_sentiment` is event-level FinBERT `fb_score`.
- `06_predictions_vs_actual_finbert.png`,
  `07_residuals_by_variant_finbert.png` — same schema as the VADER versions.
