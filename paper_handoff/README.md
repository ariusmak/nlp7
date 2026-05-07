# Paper Handoff

This folder contains the result tables needed to finish the current report
draft, `NLP Research Report.docx` dated 2026-05-07.

The report already has the broad story, data section, sentiment methodology,
model design, basic figures, and a basic test-metric table. The remaining
work is to add stronger evidence, correct a few overclaims, and make the
main conclusion harder to misread.

Main conclusion to preserve:

WallStreetBets sentiment does **not** reliably improve out-of-sample
prediction of 3-trading-day post-earnings returns. Adapted VADER gives a
tiny OLS RMSE improvement on the 75-event test set, but the improvement is
within bootstrap uncertainty and directional accuracy falls. FinBERT does
not improve OLS performance. Ridge results are essentially unchanged by
adding sentiment.

## Highest-Priority Edits

1. In Section 2, fix the event window wording.
   The report currently says posts were connected using "ten trading days
   prior to earnings." The locked design is **10 calendar days before the
   earnings date, excluding the earnings day itself**.

2. In Section 5.1, add a per-ticker coverage table.
   Use `event_sentiment_per_ticker_vader.csv`. This supports the report's
   claim that coverage is uneven across tickers.

3. In Section 5.4, add numeric correlation evidence.
   Use `eda_feature_target_correlations.csv`. The current report says the
   scatterplots look weak; this file gives the actual numbers.

4. In Section 5.4 or 7.1, add scorer disagreement evidence.
   Use `robustness_cross_scorer_event_corr.csv` and
   `sentiment_label_confusion_finbert.csv`. VADER and FinBERT both fail to
   produce useful predictive scatterplots, but they also measure very
   different sentiment constructs.

5. In Section 5.5, expand Table 2 or add supporting regression evidence.
   Use `model_metrics_vader.csv`, `model_metrics_finbert.csv`,
   `final_comparison_summary.csv`, `final_comparison_coefficients_wide.csv`,
   `robustness_joint_ftest.csv`, and
   `robustness_bootstrap_test_rmse.csv`.

6. In Section 6 / 7.1, soften the thin-coverage explanation.
   The report currently says thin coverage likely attenuates the sentiment
   effect. The `post_count >= 10` robustness check does **not** support
   that as the main explanation.

7. Remove the duplicate Bollen et al. (2011) reference.

## Model Setup To Keep Consistent

All OLS+sentiment models use `target_ret3` regressed on:

- `ret5_pre`
- `ret20_pre`
- `vol20_pre`
- `abvol_pre`
- `post_count`
- `mean_sentiment`
- `median_sentiment`
- `std_sentiment`
- `frac_positive`
- `frac_negative`
- ticker fixed effects
- calendar-year fixed effects

Train window: 2016-2021, n = 360.

Test window: 2022 through 2023-03-28, n = 75.

FinBERT is a robustness/scorer-comparison check. The primary sentiment
method remains adapted VADER.

## Section 5.1: Dataset Coverage

### `event_sentiment_per_ticker_vader.csv`

Use this as a new table immediately after Table 1.

Suggested caption:

> Per-ticker Reddit coverage and event-level adapted-VADER sentiment
> statistics, 2016 to 2023-03-28.

What it contains:

One row per ticker with:

- `events`
- `zero_events`
- `thin_events` where `post_count < 10`
- `mean_posts`
- `mean_sent`
- `mean_fracpos`
- `mean_fracneg`

What to say:

Coverage is highly uneven. TSLA averages about 207 matched posts per event.
GOOGL averages about 111. By contrast, GS, JPM, INTC, WMT, and BA have much
lower average coverage, and BA has the most thin events. This table supports
the report's discussion of uneven Reddit coverage across the 15-stock
universe.

Also note that adapted-VADER mean sentiment is positive for every ticker
(roughly +0.20 to +0.41). This is worth mentioning as a calibration issue:
WSB text scores positive under adapted VADER across the full universe.

### `event_sentiment_per_ticker_finbert.csv`

Optional appendix table or side-by-side companion to the VADER table.

What to say:

FinBERT mean sentiment is negative for every ticker, roughly -0.05 to -0.21.
This is useful because it shows that "WSB sentiment" depends heavily on the
scoring method. Adapted VADER and FinBERT do not produce the same
per-ticker sentiment picture.

## Section 5.4: Sentiment and Returns

### `eda_feature_target_correlations.csv`

Use this to add a small inline table of the top correlations, or cite it in
the paragraph around Figure 5.

What it contains:

Pearson correlations between numeric features and `target_ret3`, computed
on the well-covered subset where `post_count >= 10` (n = 322).

Important numbers:

- `frac_negative`: r = 0.103
- `frac_positive`: r = -0.063
- `std_sentiment`: r = 0.063
- `ret5_pre`: r = 0.060
- `ret20_pre`: r = 0.048

What to say:

The largest absolute correlation is only about 0.103, meaning the strongest
single feature explains only about 1.1% of target variance. The next-largest
correlations are around 0.06. This gives numeric support for the current
Figure 5 interpretation: the fitted relationship between sentiment and
post-earnings returns is almost flat.

Avoid saying the largest correlation is only 0.05; it is closer to 0.10.
The correct interpretation is still "weak."

### `robustness_cross_scorer_event_corr.csv`

Use this near the end of Section 5.4 or in Section 7.1.

What it contains:

Event-level Pearson correlations between adapted-VADER and FinBERT for the
six sentiment features, computed on 425 events with at least one matched
post.

Important numbers:

- `mean_sentiment`: r = 0.083
- `median_sentiment`: r = -0.044
- `std_sentiment`: r = 0.397
- `frac_positive`: r = 0.217
- `frac_negative`: r = 0.237

What to say:

Both VADER and FinBERT fail to produce useful predictive scatterplots, but
they also disagree strongly with each other. The mean and median sentiment
features are essentially uncorrelated across scorers. This means "Reddit
sentiment" is operationally fragile: the conclusion does not depend only on
the regression model, but also on how sentiment is defined.

Suggested sentence:

> Although both sentiment scorers produce weak relationships with returns,
> they do not measure the same event-level construct: adapted-VADER and
> FinBERT mean sentiment correlate only 0.083 across events.

## Section 5.5: Model Performance

The current Table 2 only reports test metrics. Keep it if space is tight,
but the paper will be stronger if Section 5.5 also includes train metrics,
OLS fit statistics, coefficient evidence, joint F-tests, and bootstrap
uncertainty.

### `model_metrics_vader.csv` and `model_metrics_finbert.csv`

Use these to expand Table 2.

What they contain:

Train and test RMSE, MAE, directional accuracy, selected Ridge alpha, and
sample sizes for:

- OLS baseline
- OLS + sentiment
- Ridge baseline
- Ridge + sentiment

Important numbers:

- OLS baseline test RMSE: 0.0949
- OLS + VADER test RMSE: 0.0936
- OLS + FinBERT test RMSE: 0.0952
- Ridge baseline test RMSE: 0.0948
- Ridge + VADER test RMSE: 0.0948
- Ridge + FinBERT test RMSE: 0.0946

What to say:

The VADER OLS model has a tiny RMSE improvement, but directional accuracy
falls from 0.5733 to 0.5467. FinBERT does not improve OLS. Ridge is
basically unchanged by sentiment.

Also mention the train/test gap: train RMSE is about 0.071, while test RMSE
is about 0.094-0.095. This supports the discussion that the prediction
target is noisy and hard to forecast.

### `final_comparison_summary.csv`

Use this as a fit-stat block below the main OLS table.

What it contains:

OLS R-squared, adjusted R-squared, F-statistics, F-test p-values, and
train/test metrics for baseline, VADER, and FinBERT.

Important numbers:

- OLS baseline adjusted R-squared: -0.022
- OLS + VADER adjusted R-squared: -0.004
- OLS + FinBERT adjusted R-squared: -0.023

What to say:

All adjusted R-squared values are negative. This means the models do not
provide meaningful explanatory power after accounting for model complexity.
This is stronger evidence than simply saying RMSE is similar.

### `final_comparison_coefficients_wide.csv`

Use this for a coefficient table in Section 5.5.

The main paper should show only:

- constant
- four baseline market features
- six sentiment features

Ticker and year fixed effects can go to the appendix or be summarized with
a note saying they were included.

Important coefficient results:

- VADER `frac_negative`: coefficient about +0.110, p about 0.009
- VADER `frac_positive`: coefficient about -0.072, p about 0.067
- No other VADER sentiment feature is individually significant at 5%.
- No FinBERT sentiment feature is individually significant at 5%.

Interpret carefully:

The significant VADER coefficient is `frac_negative`, and its sign is
positive. That is the opposite of a simple "bullish sentiment predicts
positive returns" story. Do **not** describe this as evidence that Reddit
bullishness predicts post-earnings returns.

### `final_comparison_coefficients_long.csv`

Same coefficient information as the wide file, but in long format with a
`sig` column:

- `***` for p < 0.01
- `**` for p < 0.05
- `*` for p < 0.10

Use whichever format is easier for building the final regression table.

### `robustness_joint_ftest.csv`

Use this immediately after the coefficient table.

What it contains:

Joint F-tests of whether all six sentiment coefficients are zero in the
OLS+sentiment models.

Important numbers:

- VADER: F(6, 330) = 1.98, p = 0.068
- FinBERT: F(6, 330) = 0.93, p = 0.471

What to say:

At the 5% level, we cannot reject that the six sentiment coefficients are
jointly zero. VADER is borderline at the 10% level, but not conventionally
significant at 5%. FinBERT clearly fails to reject.

Suggested sentence:

> We cannot reject at conventional levels the hypothesis that the six
> sentiment coefficients are jointly zero (VADER p = 0.068; FinBERT
> p = 0.471).

### `robustness_bootstrap_test_rmse.csv`

Use this to qualify the small OLS+VADER RMSE improvement.

What it contains:

Bootstrap results on the 75-event test set using B = 2000 and seed =
20240507. The reported difference is:

`RMSE(sentiment model) - RMSE(baseline model)`

Important numbers:

- VADER Delta RMSE: -0.00137
- VADER 95% CI: [-0.0035, +0.0011]
- VADER beats baseline in 86% of bootstrap iterations
- FinBERT Delta RMSE: +0.00028
- FinBERT 95% CI: [-0.0013, +0.0020]

What to say:

The point estimate slightly favors VADER, but the confidence interval
crosses zero. The improvement is too small to treat as reliable evidence
that sentiment improves prediction.

Suggested sentence:

> The point estimate favors VADER (Delta RMSE = -0.0014), but the 95%
> bootstrap interval ranges from -0.0035 to +0.0011, so the improvement is
> within test-sample uncertainty.

### `ridge_cv_detail_vader.csv` and `ridge_cv_detail_finbert.csv`

Use these only if adding detail on Ridge validation.

What they contain:

Walk-forward Ridge cross-validation details for 33 alphas from 0.001 to
100,000:

- train 2016-2018, validate 2019
- train 2016-2019, validate 2020
- train 2016-2020, validate 2021

Important numbers:

- Ridge baseline selected alpha = 100,000
- Ridge + VADER selected alpha about 56,234
- Ridge + FinBERT selected alpha about 3,162

What to say:

The selected regularization values are large, especially for baseline and
VADER. This is consistent with weak stable predictive signal in the feature
set.

## Section 6 / 7.1: Discussion and Limitations

### `robustness_geq10_posts.csv`

Use this to revise the thin-coverage limitation.

What it contains:

OLS models rerun on the well-covered subset where `post_count >= 10`
(n = 322). It reports train/test metrics, R-squared, adjusted R-squared,
and the joint F-test for the six sentiment coefficients.

Important numbers:

- VADER sentiment F-test p-value: 0.144
- FinBERT sentiment F-test p-value: 0.144
- All subset model variants still have negative adjusted R-squared.

What to say:

Thin coverage is still a limitation because low-post events produce noisy
sentiment aggregates. However, dropping thin-coverage events does **not**
make the sentiment effect significant. Therefore, do not say thin coverage
is the main reason sentiment fails. A more accurate interpretation is:

> Thin coverage adds measurement noise, but it does not appear to be the
> binding constraint; even among events with at least 10 matched posts,
> sentiment features are not jointly significant.

### `sentiment_label_confusion_finbert.csv`

Use this in Section 7.1 when discussing limits of VADER and FinBERT on WSB
text.

What it contains:

A 3-by-3 post-level label confusion matrix. Rows are adapted-VADER labels
and columns are FinBERT labels. Counts are matched post-stock rows, not
unique Reddit posts.

Important numbers:

- Total matched post-stock rows compared: 32,012
- Adapted-VADER positive rows: 21,282
- Of VADER-positive rows, 11,817 are FinBERT-negative
- Only 5,066 VADER-positive rows are also FinBERT-positive
- Of FinBERT-positive rows, about 80% are also VADER-positive

What to say:

The disagreement is asymmetric. VADER is much more permissive on the
positive side, while FinBERT labels many VADER-positive WSB posts as
negative. This supports the limitation that sentiment scoring on WSB text
is model-dependent and linguistically difficult.

## Methodology Details If Needed

The current report already covers most of this, but these exact details can
be used to tighten Sections 3 and 4.

Sentiment:

- Adapted VADER is the primary method.
- FinBERT is a robustness check.
- VADER preprocessing uses 17 phrase replacements.
- VADER preprocessing uses 7 emoji replacements.
- The custom WSB lexicon adds 52 token scores to VADER.
- Positive threshold: compound score > 0.05.
- Negative threshold: compound score < -0.05.
- FinBERT score is `p_positive - p_negative`.
- The same six event-level features are used for VADER and FinBERT.

Event-level sentiment features:

- `post_count`
- `mean_sentiment`
- `median_sentiment`
- `std_sentiment`
- `frac_positive`
- `frac_negative`

Target and features:

- Unit of observation: one ticker-earnings event.
- Final dataset: 435 events across 15 tickers.
- Target: `target_ret3`, the 3-trading-day post-earnings return.
- Because timing metadata is unavailable, all events use the fallback
  timing rule with `t0` equal to the prior trading day's close.
- Baseline features: `ret5_pre`, `ret20_pre`, `vol20_pre`, `abvol_pre`.
- Fixed effects: ticker and calendar year.
- Ridge numeric features are mean-imputed and standardized; fixed-effect
  dummies pass through unscaled.

## Small Copyediting Fixes In The Current Report

These are not result changes, but they are easy cleanup items:

- Section 2: add a space in "2016-2023.These".
- Section 2: add a space in "very small.Sampling".
- Section 2: "The final dataset have" should be "The final dataset has".
- Section 2 / 5.1: use consistent wording for "post-event link rows" versus
  "post-event pairs."
- Section 6: use "WallStreetBets" consistently, not "Wall Street bets."
- References: remove the duplicate Bollen et al. (2011) entry.
