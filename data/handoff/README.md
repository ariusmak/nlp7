# Handoff dataset

Minimum inputs to reproduce the modeling experiment with a different sentiment analyzer.

## Files

- `posts_linked_to_events_slim.parquet` — 20,700 rows. One row per (post, earnings event) link.
  Columns: `id`, `ticker`, `event_id`, `rdq`, `window_start`, `post_dt`, `title`, `selftext`, `has_real_body`.
  These are WSB posts inside each event's 10-calendar-day pre-earnings window (excluding the earnings day itself).
- `event_modeling_skeleton.parquet` — 435 rows. One row per (ticker, earnings event).
  Columns: `ticker`, `event_id`, `rdq`, `calendar_year`, `earnings_timing`, `timing_used`,
  `ret5_pre`, `ret20_pre`, `vol20_pre`, `abvol_pre`, `target_ret3`, `target_start_date`, `target_end_date`.
  The baseline market features and the target are already computed; only sentiment is missing.

## How to use

1. Score every post in `posts_linked_to_events_slim` with your sentiment analyzer. Use `title + " " + selftext`
   as the input text (same convention as the primary pipeline).
2. Aggregate per `event_id` into the six features used in the main spec:
   `post_count`, `mean_sentiment`, `median_sentiment`, `std_sentiment`, `frac_positive`, `frac_negative`.
   Positive/negative thresholds follow whatever your analyzer uses; for VADER the project uses
   `compound > 0.05` / `< -0.05`.
3. Left-join your six event-level features onto `event_modeling_skeleton` by `event_id`. Events with no
   matched posts (10 of 435) should be kept and either imputed or dropped — match whatever the main
   pipeline does.
4. Run the same OLS / Ridge training split as the main experiment:
   train 2016-2021, test 2022 through 2023-03-28, Ridge α tuned via walk-forward CV within train.

## Caveats

- Text is unmoderated WSB content. Slurs, NSFW language, and offensive posts exist in the corpus.
- Timestamps are UTC. `rdq` is a naive date; the pipeline treats it as the "day of earnings" and
  the post window is `[rdq - 10 calendar days, rdq)`.
- All 435 events land in `timing_used = "fallback_prev_close"` because WRDS `comp.fundq` does not
  carry before-open/after-close metadata.
