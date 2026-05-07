# Paper Handoff

Files for the writeup, organized in the order they should appear in the
paper. Each entry lists what's in the file, where it goes, and what the
result means so you know what to write.

The OLS+VADER and OLS+FinBERT models throughout are: `target_ret3` regressed
on `ret5_pre`, `ret20_pre`, `vol20_pre`, `abvol_pre`, the six event-level
sentiment features (`post_count`, `mean_sentiment`, `median_sentiment`,
`std_sentiment`, `frac_positive`, `frac_negative`), ticker fixed effects,
and calendar-year fixed effects. Train = 2016-2021 (n=360), test =
2022 through 2023-03-28 (n=75).

---

## Section 5.1 (Dataset Coverage) — additions

### `event_sentiment_per_ticker_vader.csv`

**What it contains.** One row per ticker (15 rows) with `events`,
`zero_events`, `thin_events` (post_count<10), `mean_posts`, `mean_sent`,
`mean_fracpos`, `mean_fracneg`. Computed on adapted-VADER scores.

**Where it goes.** New table in Section 5.1, immediately after Table 1.
Suggested caption: "Per-ticker Reddit coverage and event-level adapted-VADER
sentiment statistics (2016 to 2023-03-28)."

**Why it matters / what to write.** Section 7.1 already claims "TSLA and
NVDA are well-covered, while GS, BA, and WMT frequently fall below the
ten-post threshold" but this table is what proves it. TSLA averages ~207
matched posts/event; GS, JPM, INTC, WMT all average under ~14, and BA has
the most thin events. Use this to motivate the ≥10-post robustness check
later. Mean sentiment is positive across every ticker (range roughly
+0.20 to +0.41), which itself is worth noting as a calibration concern.

### `event_sentiment_per_ticker_finbert.csv`

**What it contains.** Same schema, computed on FinBERT `fb_score`.

**Where it goes.** Optional appendix companion table or merged into the
table above as side-by-side `mean_sent_VADER` / `mean_sent_FinBERT`
columns.

**Why it matters / what to write.** FinBERT mean sentiment is *negative*
for every ticker (range roughly -0.05 to -0.21). The two scorers are
flipping the per-ticker sign — direct evidence that "WSB sentiment" is
not a well-defined quantity, it depends on the scorer. Reinforces the
scorer-mismatch argument in Section 7.1.

---

## Section 5.4 (Sentiment and Returns) — additions

### `eda_feature_target_correlations.csv`

**What it contains.** Pearson r of every numeric feature with `target_ret3`,
sorted by `|r|`. Computed on the well-covered subset (post_count >= 10,
N=322). Includes baseline market features and the six VADER sentiment
features.

**Where it goes.** Section 5.4, just before or after Figure 5. Either as a
small inline table of the top 5 correlations or as an appendix table.

**Why it matters / what to write.** Section 5.4 currently says correlation
is "weak" without numbers. The largest |r| is roughly 0.05 - 0.06, so even
the *best* feature explains under 0.4% of variance in the target. Provides
the quantitative backing for the "fitting curve is almost flat" sentence
already in the paper.

### `robustness_cross_scorer_event_corr.csv`

**What it contains.** Event-level Pearson r between adapted-VADER and
FinBERT for each of the six sentiment features (425 events with >=1 post).

**Where it goes.** End of Section 5.4 as a one-paragraph aside, or as a
sentence in Section 6 / 7.1.

**Why it matters / what to write.** `mean_sentiment` r = 0.083,
`median_sentiment` r = -0.044. The two scorers barely correlate at the
event level; the median actually inverts. `std_sentiment` (r=0.40),
`frac_positive` (r=0.22), `frac_negative` (r=0.24) are higher but still
weak. This is the single strongest piece of evidence that "Reddit
sentiment" is operationally ill-defined: changing scorer changes the
input feature, and the two operationalizations measure essentially
orthogonal constructs at the event level.

---

## Section 5.5 (Model Performance) — core regression evidence

### `final_comparison_coefficients_wide.csv` (and `_long.csv`)

**What it contains.** OLS coefficients with p-values for the three OLS
variants — baseline, VADER+sentiment, FinBERT+sentiment — pivoted into
parallel `coef_*` and `pvalue_*` columns. Includes `const`, the four
market features, the six sentiment features, all 14 ticker fixed effects,
and all 7 year fixed effects. The `_long.csv` is the same data unpivoted
with a `sig` column (`*** p<0.01`, `** p<0.05`, `* p<0.10`).

**Where it goes.** New regression table in Section 5.5, replacing or
supplementing Table 2. The main body should show only the core rows
(constant, market features, sentiment features) with stars; ticker and
year FEs go to an appendix with a note that they were included.

**Why it matters / what to write.** Currently Section 5.5 makes claims
about coefficient signs and significance with no coefficient table to
back them up. Things you can now write:
- The four baseline market features are the same magnitude across
  variants — sentiment doesn't crowd them out.
- For VADER, the only sentiment feature that crosses p<0.05 is
  `frac_negative` (positive coefficient, ~+0.11, p~0.009) — the
  *opposite* sign from a "bullish sentiment helps" story. None of the
  other VADER features are individually significant.
- For FinBERT, none of the six sentiment features are individually
  significant.
- This justifies the joint F-test that comes next.

### `final_comparison_summary.csv`

**What it contains.** R^2, adj R^2, F-stat, F p-value, train/test
RMSE/MAE, directional accuracy for the three OLS variants.

**Where it goes.** Section 5.5, as a small fit-statistic block above
Table 2 or merged into Table 2's bottom rows.

**Why it matters / what to write.** All three models have **negative
adj R^2** (-0.024 to -0.042 depending on variant). The full-model F-test
is non-significant for both sentiment variants. This is a much sharper
finding than "RMSE is similar": adjusted for parameter count, the
sentiment-augmented models *fit worse* than the baseline.

### `model_metrics_vader.csv` and `model_metrics_finbert.csv`

**What it contains.** Train and test RMSE, MAE, directional accuracy, plus
selected ridge alpha and `n` for all four model variants per scorer (OLS
baseline, OLS+sentiment, Ridge baseline, Ridge+sentiment).

**Where it goes.** Section 5.5 — replace Table 2 with a version that
shows train metrics alongside test metrics.

**Why it matters / what to write.** The current Table 2 only shows test
metrics, which hides the train-test gap. Train RMSE is ~0.071; test RMSE
is ~0.094-0.095. That ~33% inflation is the standard overfitting signal
and supports the "noisy target" narrative in Section 6.

### `ridge_cv_detail_vader.csv` and `ridge_cv_detail_finbert.csv`

**What it contains.** Walk-forward CV grid: every (alpha, val_year) pair
with its RMSE/MAE/dir_acc/n, plus the cross-fold `cv_rmse` per alpha,
tagged by variant. 33 alphas in `np.logspace(-3, 5, 33)`, 3 folds, 2
variants per file.

**Where it goes.** Section 4 (Model Design) — describe the alpha grid and
fold structure; cite the file as appendix. Optionally a single sentence
in Section 5.5.

**Why it matters / what to write.** Both VADER ridge_sentiment and ridge
baseline pick alpha = 100,000 (the **top** of the grid) — the regularizer
is shrinking sentiment to zero by choice. FinBERT ridge_sentiment picks
alpha = 3,162. Worth one sentence in Section 5.5: "The Ridge cross-
validation selected alpha at the top of the search grid for both
baseline and sentiment-augmented variants, which is consistent with the
OLS finding that the additional sentiment features carry no
predictive load."

### `robustness_joint_ftest.csv` (item 8 from earlier discussion)

**What it contains.** Joint F-test that all six sentiment coefficients are
zero, for both OLS+VADER and OLS+FinBERT on the full sample. Reports F,
df_num, df_denom, p-value, and the unrestricted-model R^2 / adj R^2.

**Where it goes.** Section 5.5, immediately after the coefficient table.
Two-sentence paragraph.

**Why it matters / what to write.** This is the formal version of the
paper's "no improvement" claim.
- VADER: F(6, 330) = 1.98, **p = 0.068**. Borderline — does not reject at
  α=0.05 but does reject at α=0.10.
- FinBERT: F(6, 330) = 0.93, **p = 0.47**. Solidly fails to reject.

Phrasing: "We cannot reject at conventional levels the hypothesis that
the six sentiment coefficients are jointly zero (VADER p=0.068, FinBERT
p=0.47)."

### `robustness_bootstrap_test_rmse.csv` (item 9)

**What it contains.** Bootstrap on the 75-event test set with B=2000 and
seed=20240507. Reports the point estimate of the RMSE difference
(sentiment - baseline), the 95% percentile CI, and the share of
bootstrap iterations in which sentiment beats baseline.

**Where it goes.** Section 5.5, paired with the paragraph above. One
sentence per scorer.

**Why it matters / what to write.**
- VADER: ΔRMSE = -0.00137, 95% CI **[-0.0035, +0.0011]** (straddles 0),
  86% of bootstrap iterations show VADER beating baseline.
- FinBERT: ΔRMSE = +0.00028, 95% CI [-0.0013, +0.0020] (straddles 0,
  37% below zero).

Replaces the loose "0.0936 vs 0.0949" claim with a real CI. Suggested
phrasing: "The point estimate favors VADER (ΔRMSE = -0.0014), but the
95% percentile bootstrap CI on the 75-event test set ranges from -0.0035
to +0.0011 — within sampling noise."

---

## Section 6 / 7.1 (Discussion / Limitations) — interpretive evidence

### `robustness_geq10_posts.csv` (item 10)

**What it contains.** OLS rerun on the 322-event well-covered subset
(events with `post_count >= 10`) for both VADER and FinBERT. Reports
train+test metrics, R^2, adj R^2, and the joint F-test that the six
sentiment coefficients are zero on this subset.

**Where it goes.** Section 6 (Discussion) or Section 7.1 (Limitations) —
this is the empirical check on the thin-coverage hypothesis.

**Why it matters / what to write.** **This actually weakens the thin-
coverage explanation.** On the well-covered subset:
- VADER: sent F p-value = **0.144** (worse than 0.068 on the full sample)
- FinBERT: sent F p-value = **0.144** (better than 0.47 but still
  non-significant)
- All four subset variants still have negative adj R^2.

Rephrase Section 7.1's thin-coverage paragraph: do not claim thin coverage
attenuates the sentiment effect, because the F-stat does *not* improve
when those events are dropped. The honest reading is that thin coverage
adds noise but is not the binding constraint — even the well-covered
subset shows no significant sentiment effect.

### `sentiment_label_confusion_finbert.csv` (item 5)

**What it contains.** 3x3 confusion matrix of post-level labels. Rows =
adapted-VADER label (negative / neutral / positive). Columns = FinBERT
label. Cells are post counts. Computed on all ~20,640 scored posts that
appear in both files.

**Where it goes.** Section 7.1 (Limitations), supporting the "VADER and
FinBERT have structural limits on WSB text" paragraph.

**Why it matters / what to write.** The two scorers disagree
catastrophically. Of the 21,282 posts adapted-VADER labels positive,
**11,817 (55%) are labeled negative by FinBERT**, and only 5,066 (24%)
agree. Of FinBERT-positive posts, ~76% (5,066 / 6,341) are also
VADER-positive, but the asymmetry shows VADER is much more permissive
on the positive side. Use this as the strongest empirical anchor for
the scorer-limitations paragraph; pair it with item 11 (cross-scorer
event-level r) for the same point at the aggregated level.

---

## Section 3 (Sentiment Methodology) — content to write

These are not file artifacts; they are configuration values from the
locked design (`configs/sentiment.yaml`, `configs/wsb_*.json`,
`scripts/10_train_models.py`). Use them verbatim or paraphrase.

- **Two sentiment scorers compared:** VADER (rule-based, with custom WSB
  lexicon extension) and FinBERT (`ProsusAI/finbert`, BERT-based, trained
  on financial news, 3-class softmax over {positive, neutral, negative}).
- **VADER preprocessing pipeline (deterministic, applied before scoring):**
  1. Multi-word phrase replacement — **17 phrases** (e.g. `to the moon`
     -> `to_the_moon`, `diamond hands` -> `diamond_hands`, `short squeeze`
     -> `short_squeeze`, `paper hands` -> `paper_hands`, `dead cat bounce`
     -> `dead_cat_bounce`).
  2. Emoji-to-token replacement — **7 emoji** (rocket -> `rocket`,
     diamond -> `diamond`, diamond + raised-hands -> `diamond_hands`,
     bear -> `bear`, ape -> `ape`, chart-up -> `going_up`,
     chart-down -> `going_down`).
  3. Lexicon update — **52 custom WSB tokens** appended to VADER's default
     lexicon via `analyzer.lexicon.update()`.
- **WSB lexicon design choices worth noting in prose:**
  - Strong-positive (score 2.5 to 3.0): `to_the_moon`, `rocket`,
    `tendies`, `diamond_hands`, `all_time_high`, `mooning`.
  - Strong-negative (score -2.5 to -3.0): `bagholder`, `paper_hands`,
    `rug_pull`, `dead_cat_bounce`, `sea_of_red`.
  - **Self-deprecating tokens overridden to 0.0** to prevent VADER from
    mis-scoring community in-group language as strongly negative:
    `retard`, `retarded`, `autist`, `autists`, `smooth_brain`. This
    override is the most consequential single edit; without it a large
    share of WSB posts would flip from neutral to strongly negative.
  - Trading jargon (`calls`, `puts`, `dd`, `yolo`) explicitly **left at
    0.0** — they are directional bets, not sentiment.
- **FinBERT scoring:** batch size 32, max sequence length 512 tokens,
  scored on `title + " " + selftext` (same input as VADER). Compound score
  defined as `fb_score = p_pos - p_neg in [-1, 1]`.
- **Compound thresholds (both scorers):** positive if compound > **0.05**,
  negative if compound < **-0.05**, neutral otherwise.
- **Event-level aggregation:** for each (ticker, earnings event), six
  features are computed over posts in the 10-calendar-day pre-earnings
  window — `post_count`, `mean_sentiment`, `median_sentiment`,
  `std_sentiment` (sample std, ddof=1), `frac_positive`, `frac_negative`.
  Events with zero matched posts are kept (10 of 435) and the five
  sentiment columns are mean-imputed at training time.

---

## Section 4 (Model Design) — content to write

- **Unit of observation:** one (ticker, earnings event) pair. Final
  dataset = 435 events across 15 tickers, 2016-01-01 through 2023-03-28.
- **Target:** `target_ret3`, the 3-trading-day post-earnings return
  defined as `(close_t3 - close_t0) / close_t0`. Because WRDS Compustat
  does not carry before-open / after-close timing flags, all events use
  the **fallback timing rule**: `t0` = the prior trading day's close.
- **Baseline market features (4):** `ret5_pre`, `ret20_pre` (5- and
  20-trading-day returns ending on the day before the event), `vol20_pre`
  (20-day realized volatility of daily returns), `abvol_pre`
  (day-minus-1 volume / trailing 20-day mean volume).
- **Sentiment features (6):** `post_count`, `mean_sentiment`,
  `median_sentiment`, `std_sentiment`, `frac_positive`, `frac_negative` —
  same definitions for VADER and FinBERT, but sourced from each scorer's
  compound.
- **Fixed effects:** ticker (15 levels, 14 dummies after dropping
  reference) and calendar year (8 levels 2016 to 2023, 7 dummies after
  dropping reference). Encoded via `pandas.get_dummies(..., drop_first=True)`.
- **Models compared:** 4 variants per scorer — OLS-baseline,
  OLS+sentiment, Ridge-baseline, Ridge+sentiment.
- **Train/test split (time-respecting, no random shuffle):** train =
  2016 to 2021 (n=360), test = 2022 through 2023-03-28 (n=75).
- **Ridge alpha selection (walk-forward CV inside train):** three
  expanding folds — train 2016 to 2018 / val 2019, train 2016 to 2019 /
  val 2020, train 2016 to 2020 / val 2021. Search grid:
  `np.logspace(-3, 5, 33)` — **33 alphas from 0.001 to 100,000**.
  Cross-fold score is sample-size-weighted RMSE; the alpha minimizing it
  is refit on the full 2016 to 2021 train window before scoring the
  holdout.
- **Standardization (Ridge only):** numeric features (baseline + sentiment)
  are mean-imputed and standardized; ticker/year dummies pass through
  unscaled.
- **Evaluation metrics:** RMSE, MAE, directional accuracy
  (sign(y_pred) == sign(y_true)). Reported on both train and test for
  transparency.
