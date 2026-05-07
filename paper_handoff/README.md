# Paper Handoff

This folder is for finishing the current report draft,
`NLP Research Report.docx` dated 2026-05-07.

The CSVs are kept for audit/exact decimals, but teammates should be able to
use the trimmed tables below directly in the paper. The report already has
the broad narrative, data section, sentiment methodology, model design,
basic figures, and a basic test-metric table. The remaining work is to add
the stronger evidence below and fix a few overclaims.

Main conclusion to preserve:

WallStreetBets sentiment does **not** reliably improve out-of-sample
prediction of 3-trading-day post-earnings returns. Adapted VADER gives a
tiny OLS RMSE improvement on the 75-event test set, but the improvement is
within bootstrap uncertainty and directional accuracy falls. FinBERT does
not improve OLS performance. Ridge results are essentially unchanged by
adding sentiment.

## Highest-Priority Edits

1. Section 2: change "ten trading days prior to earnings" to **10 calendar
   days before the earnings date, excluding the earnings day itself**.
2. Section 5.1: add the per-ticker coverage table below.
3. Section 5.4: add the feature/target correlation table below.
4. Section 5.4 or 7.1: add the scorer-disagreement evidence below.
5. Section 5.5: expand Table 2 or add supporting regression tables using
   the model-performance, fit-stat, coefficient, F-test, and bootstrap
   tables below.
6. Section 6 / 7.1: soften the thin-coverage explanation. The well-covered
   subset check does **not** show that thin coverage is the main reason the
   sentiment effect fails.
7. References: remove the duplicate Bollen et al. (2011) entry.

## Model Setup To Keep Consistent

All OLS+sentiment models use `target_ret3` regressed on:

- baseline features: `ret5_pre`, `ret20_pre`, `vol20_pre`, `abvol_pre`
- sentiment features: `post_count`, `mean_sentiment`, `median_sentiment`,
  `std_sentiment`, `frac_positive`, `frac_negative`
- ticker fixed effects
- calendar-year fixed effects

Train window: 2016-2021, n = 360.

Test window: 2022 through 2023-03-28, n = 75.

FinBERT is a robustness/scorer-comparison check. The primary sentiment
method remains adapted VADER.

## Section 5.1: Dataset Coverage

Use this as a new table immediately after Table 1.

Suggested caption:

> Per-ticker Reddit coverage and event-level sentiment statistics,
> 2016 to 2023-03-28.

Source files:

- `event_sentiment_per_ticker_vader.csv`
- `event_sentiment_per_ticker_finbert.csv`

| Ticker | Events | Zero Events | Thin Events | Mean Posts | Mean Sent VADER | Mean Sent FinBERT |
|---|---:|---:|---:|---:|---:|---:|
| TSLA | 29 | 0 | 0 | 206.8 | 0.200 | -0.108 |
| GOOGL | 29 | 0 | 0 | 110.9 | 0.279 | -0.107 |
| AMZN | 29 | 0 | 3 | 73.7 | 0.337 | -0.109 |
| AAPL | 29 | 0 | 0 | 61.2 | 0.259 | -0.122 |
| MSFT | 29 | 0 | 7 | 35.4 | 0.324 | -0.154 |
| NVDA | 29 | 0 | 3 | 35.2 | 0.267 | -0.069 |
| MU | 29 | 2 | 13 | 29.8 | 0.263 | -0.111 |
| AMD | 29 | 0 | 4 | 27.4 | 0.300 | -0.108 |
| NFLX | 29 | 0 | 6 | 27.3 | 0.253 | -0.126 |
| DIS | 29 | 1 | 11 | 24.9 | 0.319 | -0.129 |
| BA | 29 | 3 | 18 | 19.5 | 0.304 | -0.166 |
| WMT | 29 | 1 | 13 | 18.2 | 0.281 | -0.072 |
| INTC | 29 | 2 | 13 | 16.6 | 0.302 | -0.076 |
| JPM | 29 | 0 | 9 | 15.2 | 0.421 | -0.183 |
| GS | 29 | 1 | 13 | 11.7 | 0.383 | -0.046 |

What to write:

Coverage is highly uneven. TSLA averages about 207 matched posts per event,
while GOOGL averages about 111. Several tickers have much thinner coverage:
GS, JPM, INTC, WMT, and BA all average far fewer posts per event, and BA
has the most thin events. This supports the report's claim that Reddit
coverage is concentrated in a subset of retail-favorite tickers.

Also note the scorer contrast. Adapted-VADER mean sentiment is positive for
every ticker, while FinBERT mean sentiment is negative for every ticker.
This is useful evidence that "WSB sentiment" depends heavily on the scoring
method.

## Section 5.4: Sentiment and Returns

### Feature/Target Correlations

Use this around Figure 5 to put numbers behind the "weak relationship"
claim.

Source file: `eda_feature_target_correlations.csv`

Computed on the well-covered subset where `post_count >= 10` (n = 322).

| Feature | Pearson r with `target_ret3` |
|---|---:|
| `frac_negative` | 0.103 |
| `frac_positive` | -0.063 |
| `std_sentiment` | 0.063 |
| `ret5_pre` | 0.060 |
| `ret20_pre` | 0.048 |
| `vol20_pre` | 0.048 |
| `mean_sentiment` | -0.036 |
| `abvol_pre` | 0.034 |
| `median_sentiment` | -0.020 |
| `post_count` | 0.000 |

What to write:

The largest absolute correlation is only about 0.103, meaning the strongest
single feature explains only about 1.1% of target variance. The next-largest
correlations are around 0.06. This numerically supports the visual result in
Figure 5: sentiment and post-earnings returns have a very weak relationship.

Avoid saying the largest correlation is only 0.05; it is closer to 0.10.
The correct interpretation is still "weak."

### Cross-Scorer Agreement

Use this near the end of Section 5.4 or in Section 7.1.

Source file: `robustness_cross_scorer_event_corr.csv`

Computed on 425 events with at least one matched post.

| Sentiment Feature | VADER-FinBERT Event-Level r |
|---|---:|
| `post_count` | 1.000 |
| `mean_sentiment` | 0.083 |
| `median_sentiment` | -0.044 |
| `std_sentiment` | 0.397 |
| `frac_positive` | 0.217 |
| `frac_negative` | 0.237 |

What to write:

Both VADER and FinBERT fail to produce useful predictive scatterplots, but
they also disagree strongly with each other. Mean and median sentiment are
essentially uncorrelated across scorers. This means "Reddit sentiment" is
operationally fragile: the conclusion depends not only on the regression
model, but also on how sentiment is defined.

Suggested sentence:

> Although both sentiment scorers produce weak relationships with returns,
> they do not measure the same event-level construct: adapted-VADER and
> FinBERT mean sentiment correlate only 0.083 across events.

## Section 5.5: Model Performance

The current Table 2 only reports test metrics. Keep it if space is tight,
but the paper will be stronger if Section 5.5 also includes train metrics,
OLS fit statistics, coefficient evidence, joint F-tests, and bootstrap
uncertainty.

### Expanded Model Metrics

Use this to replace or expand Table 2.

Source files:

- `model_metrics_vader.csv`
- `model_metrics_finbert.csv`

| Model | Sentiment Source | Alpha | Train RMSE | Test RMSE | Test MAE | Test Directional Accuracy |
|---|---|---:|---:|---:|---:|---:|
| OLS baseline | None |  | 0.0715 | 0.0949 | 0.0687 | 0.5733 |
| OLS + VADER | Adapted VADER |  | 0.0702 | 0.0936 | 0.0677 | 0.5467 |
| OLS + FinBERT | FinBERT |  | 0.0709 | 0.0952 | 0.0689 | 0.5733 |
| Ridge baseline | None | 100000 | 0.0731 | 0.0948 | 0.0701 | 0.5333 |
| Ridge + VADER | Adapted VADER | 56234 | 0.0731 | 0.0948 | 0.0701 | 0.5333 |
| Ridge + FinBERT | FinBERT | 3162 | 0.0729 | 0.0946 | 0.0698 | 0.5333 |

What to write:

The VADER OLS model has a tiny RMSE improvement, but directional accuracy
falls from 0.5733 to 0.5467. FinBERT does not improve OLS. Ridge is
basically unchanged by sentiment.

Also mention the train/test gap: train RMSE is about 0.071, while test RMSE
is about 0.094-0.095. This supports the discussion that the prediction
target is noisy and hard to forecast.

### OLS Fit Statistics

Use this as a fit-stat block below the OLS regression table.

Source file: `final_comparison_summary.csv`

| OLS Variant | R-squared | Adjusted R-squared | Full-Model F p-value | Train RMSE | Test RMSE | Test Directional Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.043 | -0.022 | 0.880 | 0.0715 | 0.0949 | 0.5733 |
| + VADER | 0.077 | -0.004 | 0.551 | 0.0702 | 0.0936 | 0.5467 |
| + FinBERT | 0.059 | -0.023 | 0.858 | 0.0709 | 0.0952 | 0.5733 |

What to write:

All adjusted R-squared values are negative. This means the OLS models do
not provide meaningful explanatory power after accounting for model
complexity. This is stronger evidence than simply saying RMSE is similar.

### Sentiment Coefficients

Use this as the core coefficient evidence. It is better to show only the
six sentiment rows in the main text and state that market controls, ticker
fixed effects, and year fixed effects are included.

Source files:

- `final_comparison_coefficients_wide.csv`
- `final_comparison_coefficients_long.csv`

| Feature | VADER Coef. | VADER p-value | FinBERT Coef. | FinBERT p-value |
|---|---:|---:|---:|---:|
| `post_count` | -0.0000 | 0.985 | -0.0000 | 0.866 |
| `mean_sentiment` | 0.068 | 0.207 | 0.067 | 0.186 |
| `median_sentiment` | 0.020 | 0.487 | -0.072 | 0.062 |
| `std_sentiment` | -0.0000 | 1.000 | 0.009 | 0.770 |
| `frac_positive` | -0.072 | 0.067 | 0.013 | 0.663 |
| `frac_negative` | 0.110 | 0.009 | 0.021 | 0.449 |

Interpret carefully:

The only VADER sentiment feature significant at 5% is `frac_negative`, and
its coefficient is positive. That is the opposite of a simple "bullish
sentiment predicts positive returns" story. Do **not** describe this as
evidence that Reddit bullishness predicts post-earnings returns.

For FinBERT, none of the six sentiment features are significant at 5%.

### Joint F-Test For Sentiment Features

Use this immediately after the coefficient table.

Source file: `robustness_joint_ftest.csv`

| Model | F-test | p-value | Interpretation |
|---|---:|---:|---|
| OLS + VADER | 1.979 | 0.068 | Borderline at 10%, not significant at 5% |
| OLS + FinBERT | 0.934 | 0.471 | Not significant |

What to write:

At the 5% level, we cannot reject that the six sentiment coefficients are
jointly zero. VADER is borderline at the 10% level, but not conventionally
significant at 5%. FinBERT clearly fails to reject.

Suggested sentence:

> We cannot reject at conventional levels the hypothesis that the six
> sentiment coefficients are jointly zero (VADER p = 0.068; FinBERT
> p = 0.471).

### Bootstrap Test-Set RMSE Uncertainty

Use this to qualify the small OLS+VADER RMSE improvement.

Source file: `robustness_bootstrap_test_rmse.csv`

Reported difference is:

`RMSE(sentiment model) - RMSE(baseline model)`

| Comparison | Baseline RMSE | Sentiment RMSE | RMSE Difference | 95% CI | Share Sentiment Beats Baseline |
|---|---:|---:|---:|---|---:|
| OLS + VADER vs baseline | 0.0949 | 0.0936 | -0.0014 | [-0.0035, +0.0011] | 0.861 |
| OLS + FinBERT vs baseline | 0.0949 | 0.0952 | +0.0003 | [-0.0013, +0.0020] | 0.372 |

What to write:

The point estimate slightly favors VADER, but the confidence interval
crosses zero. The improvement is too small to treat as reliable evidence
that sentiment improves prediction.

Suggested sentence:

> The point estimate favors VADER (RMSE difference = -0.0014), but the 95%
> bootstrap interval ranges from -0.0035 to +0.0011, so the improvement is
> within test-sample uncertainty.

### Ridge Cross-Validation Detail

Use this only if adding detail on Ridge validation.

Source files:

- `ridge_cv_detail_vader.csv`
- `ridge_cv_detail_finbert.csv`

| Ridge Variant | Selected Alpha |
|---|---:|
| Ridge baseline | 100000 |
| Ridge + VADER | 56234 |
| Ridge + FinBERT | 3162 |

What to write:

The selected regularization values are large, especially for baseline and
VADER. This is consistent with weak stable predictive signal in the feature
set.

## Section 6 / 7.1: Discussion and Limitations

### Well-Covered Subset Check

Use this to revise the thin-coverage limitation.

Source file: `robustness_geq10_posts.csv`

Subset: events where `post_count >= 10`, n = 322.

| Variant | Train n | Test n | Test RMSE | Test Directional Accuracy | Adjusted R-squared | Sentiment F p-value |
|---|---:|---:|---:|---:|---:|---:|
| VADER baseline | 263 | 59 | 0.0994 | 0.593 | -0.041 |  |
| VADER + sentiment | 263 | 59 | 0.0997 | 0.542 | -0.025 | 0.144 |
| FinBERT baseline | 263 | 59 | 0.0994 | 0.593 | -0.041 |  |
| FinBERT + sentiment | 263 | 59 | 0.1029 | 0.492 | -0.025 | 0.144 |

What to write:

Thin coverage is still a limitation because low-post events produce noisy
sentiment aggregates. However, dropping thin-coverage events does **not**
make the sentiment effect significant. Therefore, do not say thin coverage
is the main reason sentiment fails.

Suggested sentence:

> Thin coverage adds measurement noise, but it does not appear to be the
> binding constraint; even among events with at least 10 matched posts,
> sentiment features are not jointly significant.

### VADER vs FinBERT Label Confusion

Use this in Section 7.1 when discussing limits of VADER and FinBERT on WSB
text.

Source file: `sentiment_label_confusion_finbert.csv`

Rows are adapted-VADER labels. Columns are FinBERT labels. Counts are
matched post-stock rows, not unique Reddit posts.

| Adapted-VADER Label | FinBERT Negative | FinBERT Neutral | FinBERT Positive |
|---|---:|---:|---:|
| Negative | 3,205 | 879 | 360 |
| Neutral | 2,547 | 2,824 | 915 |
| Positive | 11,817 | 4,399 | 5,066 |

Important numbers:

- Total matched post-stock rows compared: 32,012
- Adapted-VADER positive rows: 21,282
- Of VADER-positive rows, 11,817 are FinBERT-negative
- Only 5,066 VADER-positive rows are also FinBERT-positive
- Of FinBERT-positive rows, about 80% are also VADER-positive

What to write:

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

- Section 2: add a space in "2016-2023.These".
- Section 2: add a space in "very small.Sampling".
- Section 2: "The final dataset have" should be "The final dataset has".
- Section 2 / 5.1: use consistent wording for "post-event link rows" versus
  "post-event pairs."
- Section 6: use "WallStreetBets" consistently, not "Wall Street bets."
- References: remove the duplicate Bollen et al. (2011) entry.
