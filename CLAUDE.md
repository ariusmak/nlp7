# CLAUDE.md

## Role and objective

You are the IDE coding agent working inside the repository `nlp`.

Your job is to build a clean, reproducible, event-level NLP + finance pipeline that tests whether **WallStreetBets Reddit sentiment improves prediction of 3-trading-day post-earnings stock returns** relative to a standard baseline model.

The project is a **class project**, not a production trading system. Priorities are:

1. correctness
2. reproducibility
3. transparent methodology
4. modular code structure
5. easy handoff to human teammates for writeup

Do not optimize for cleverness. Optimize for clarity, auditability, and faithful implementation of the locked design below.

---

## Non-negotiable working style

Follow these rules throughout the repo:

- **Do not change locked project decisions without explicit user approval.**
- **Do not silently swap stocks, change formulas, change sample windows, or change matching rules.**
- Put **reusable code in `utils/`**.
- Put **one-off execution scripts in `scripts/`**.
- Use notebooks only for **EDA and result presentation**, not for core pipeline logic.
- Each script should be runnable independently from the repo root.
- All intermediate outputs should be saved to disk with stable filenames.
- Every transformation step should be inspectable.
- Prefer plain pandas/statsmodels/sklearn solutions over unnecessarily advanced tooling.
- Write code that is explicit and easy for a graduate student teammate to understand.

If any implementation detail is truly ambiguous, stop and surface it clearly rather than making a hidden decision.

---

## Research question

Does adding **WallStreetBets Reddit sentiment** to a standard pre-earnings feature set improve prediction of **3-trading-day post-earnings returns** for a fixed stock universe over the usable Reddit sample window?

---

## Locked project design

### Unit of observation

One row equals **one stock-earnings event**.

Primary key should be something like:

- `ticker`
- `earnings_date`

If timing is available, keep it in a separate field:

- `earnings_timing` in `{before_open, after_close, unknown}`

---

### Usable time window

The Reddit source is `data/raw/reddit/p4p_reddit_posts.parquet` (2,275,310 posts, converted from `p4p_reddit_posts.sql`). It spans:

- earliest Reddit timestamp: `2012-04-11 16:40:40 UTC`
- latest Reddit timestamp: `2023-03-28 22:11:08 UTC`

Use a **safe earnings-event window** based on the 10-calendar-day pre-earnings Reddit window:

- **safe start date**: `2016-01-01`
- **safe end date**: `2023-03-28`

The start was narrowed from 2012-04-19 to 2016-01-01 after EDA revealed that pre-2016 events had ~79% zero-post coverage (WSB subscriber base was too small). Under the current window (10-day lookback + expanded alias/finance-cue matching), 97.7% of 435 earnings events have ≥1 matched post.

Filter earnings events to that safe window.

Do not move these boundaries without explicit approval.

---

### Locked stock universe

The locked 15-stock universe is:

- `AAPL` — Apple
- `AMD` — Advanced Micro Devices *(ambiguous)*
- `AMZN` — Amazon
- `BA` — Boeing *(ambiguous)*
- `DIS` — Disney *(ambiguous)*
- `GOOGL` — Alphabet / Google
- `GS` — Goldman Sachs *(ambiguous)*
- `INTC` — Intel
- `JPM` — JPMorgan Chase
- `MSFT` — Microsoft
- `MU` — Micron *(ambiguous)*
- `NFLX` — Netflix
- `NVDA` — NVIDIA
- `TSLA` — Tesla
- `WMT` — Walmart

Ambiguous-flagged tickers (AMD, BA, DIS, GS, MU) require cashtag / alias / finance-cue match per the matching rules below.

Universe selection history: the original working list included BAC, MCD, QCOM, and NKE; these were swapped for BA, MU, GS (and an interim NKE attempt) after coverage EDA showed every low-volume ticker more than doubled under the p4p source with consistent (non-spike) mention patterns and every ticker cleared the 1,000-match floor in the locked window.

Important:

- Do **not** automatically replace or drop stocks.
- If coverage or false-positive issues arise, summarize the issue and surface it to the user before altering.

---

### Earnings dates source

Use **WRDS** historical earnings dates.

Expected preferred source order:

1. WRDS / Compustat quarterly earnings report date (`rdq`) if available
2. timing metadata if available from the accessible source
3. local WRDS export if direct WRDS access is not configured

Build the repo to support either:

- direct WRDS pull, or
- loading WRDS-exported CSV/Parquet from `data/raw/wrds/`

Do not hard-code a single access path.

---

### Target variable

Target is **3-trading-day post-earnings return**.

Let:

- `d` = earnings date
- `t0` = starting trading day for return window
- `t3` = third trading day after `t0`

Return formula:

\[
Y_{i,e} = \frac{P^{close}_{i,t3} - P^{close}_{i,t0}}{P^{close}_{i,t0}}
\]

#### Timing rule for `t0`

- if earnings are **before market open**, start from the **earnings-day close**
- if earnings are **after market close**, start from the **next trading day close**
- if timing is **missing**, use the **previous trading day close** as fallback

Implement this exactly.

Keep a flag indicating whether fallback timing logic was used.

---

### Reddit event window

For each stock-earnings event, include Reddit posts in the **10 calendar days before the earnings date**, excluding the earnings day itself.

Use:

- post timestamp/date in `[earnings_date - 10 calendar days, earnings_date)`

Do **not** do same-day pre-announcement filtering by exact hour.

The window was widened from 7 to 10 days after coverage EDA on the 15-ticker universe showed low-volume tickers (GS, BA, WMT, INTC, JPM) had thin-coverage rates that hurt event-level sentiment aggregation. The widened window retains pre-earnings focus while pulling more posts into low-volume events.

---

### Text used for sentiment

Use only:

- `title`
- `selftext` / body text

Do not include comments in v1.

Drop empty / deleted / removed text where needed.

Create a single combined text field such as:

- `text_for_sentiment = title + " " + selftext`

---

### Post-to-stock matching rule

This is locked.

#### Match hierarchy

A Reddit post can match a stock through:

1. **cashtag** match, e.g. `$TSLA`
2. **exact ticker token** match, e.g. `TSLA`
3. **company-name alias** match, e.g. `Tesla`, `Apple`, `Google`, `Nvidia`

Use regex token boundaries. Do not use naive substring matching.

#### Ambiguous ticker handling

Treat these as relatively more ambiguous and require stricter matching logic:

- `AMD`
- `BA`
- `DIS`
- `GS`
- `MU`
- any short/common token that produces false positives in audit logs

For ambiguous tickers, require one of:

- cashtag form, or
- company-name alias, or
- ticker plus nearby finance cue such as:
  - `stock`
  - `shares`
  - `earnings`
  - `calls`
  - `puts`
  - `guidance`
  - `bullish`
  - `bearish`
  - `buy`
  - `sell`

Implement the matching code in a way that is auditable and testable.

#### Multi-stock posts

If a post mentions multiple valid stocks:

- assign the post to **all matched stocks**
- create a `multi_ticker_flag = 1`

Also keep the full matched ticker list per post if practical.

---

### Sentiment method

Primary sentiment method is:

- **VADER compound sentiment**
- with a **custom WallStreetBets lexicon extension**
- after **deterministic phrase and emoji preprocessing**

Do not make FinBERT the primary method.

FinBERT can be added later as a robustness check, not a replacement.

---

### VADER domain adaptation design

Important design note:

Do **not** frame the main approach as free-form "translation" of Reddit language.
The primary adaptation mechanism is:

1. preprocess multi-word WSB phrases and emojis into stable tokens
2. update the VADER analyzer lexicon with WSB-specific token scores
3. score the processed text with standard VADER compound sentiment

Implement this using VADER's lexicon update mechanism, conceptually like:

```python
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(custom_wsb_lexicon)
```

The goal is to teach VADER domain vocabulary, not to rewrite entire posts.

---

### Required preprocessing before VADER

Implement a transparent, deterministic preprocessing layer that runs **after stock matching** and **before sentiment scoring**.

Required preprocessing capabilities:

1. normalize text for phrase replacement on a working copy
2. convert selected emojis into explicit tokens
3. replace selected multi-word WSB phrases with underscore-joined tokens
4. preserve the original raw text in the dataset for auditability

Examples:

- `to the moon` -> `to_the_moon`
- `short squeeze` -> `short_squeeze`
- `gamma squeeze` -> `gamma_squeeze`
- `diamond hands` -> `diamond_hands`
- `hold the line` -> `hold_the_line`
- `all time high` -> `all_time_high`

Emoji examples:

- `🚀` -> `rocket`
- `💎` -> `diamond`
- `💎🙌` -> `diamond_hands`
- `🐻` -> `bear`
- `🦍` -> `ape`
- `📈` -> `going_up`
- `📉` -> `going_down`

Important ordering rule:

- perform **post-to-stock matching first** on lightly cleaned/raw text
- perform sentiment preprocessing second

Do not let sentiment preprocessing interfere with ticker detection.

---

### WSB lexicon extension

Create explicit config files such as:

- `configs/wsb_vader_lexicon.json`
- `configs/wsb_phrase_map.json`
- optionally `configs/wsb_emoji_map.json`

Do not hard-code the full lexicon or phrase map inside modeling code.

The custom lexicon should map tokens to VADER valence scores in the standard VADER style.

Example conceptual structure:

```json
{
  "to_the_moon": 3.0,
  "rocket": 3.0,
  "diamond_hands": 2.5,
  "paper_hands": -2.0,
  "bagholder": -3.0
}
```

---

### Lexicon design rules

Use a **conservative, auditable** scoring policy.

#### Keep as clear sentiment terms

These are good candidates for the custom sentiment lexicon:

- `to_the_moon`
- `rocket`
- `tendies`
- `diamond_hands`
- `paper_hands`
- `bagholder`
- `rug_pull`
- `dead_cat_bounce`
- `red_day`
- `sea_of_red`
- `all_time_high`
- `mooning`

#### Use milder scores for stance/meme terms

Terms like these should generally get **milder** scores than extreme sentiment tokens:

- `printing`
- `ripping`
- `yeet`
- `stonks`
- `ape`
- `apes`
- `apes_together_strong`

Do not overstate these unless empirical inspection strongly supports it.

#### Do not treat all trading jargon as sentiment

These terms are risky as sentiment and should default to **0.0** or be handled as optional auxiliary text features instead of sentiment:

- `calls`
- `puts`
- `fds`
- `dd`
- `yolo`

If these are retained, document clearly that they are neutral lexical overrides, not positive/negative sentiment signals.

#### Context-dependent squeeze terms

Treat squeeze language cautiously.

- `short_squeeze` and `gamma_squeeze` may be mild positive sentiment or optional auxiliary features
- plain `squeeze` is too context-dependent to assume strongly bullish by default

Do not assign strong bullish scores to these without a documented justification.

#### Critical WSB self-identifier overrides

Override terms that VADER may otherwise score as highly negative, but that are often self-deprecating or community-neutral on WSB.

Examples include:

- `retard`
- `retarded`
- `autist`
- `autists`
- `smooth_brain`

Default recommendation:

- set these to `0.0` in the custom lexicon unless later human review suggests otherwise

This override is important because otherwise many WSB posts can be spuriously classified as strongly negative.

---

### Sentiment config and audit requirements

The sentiment subsystem must be transparent.

Required artifacts:

- saved custom lexicon file
- saved phrase replacement map
- saved emoji map if used
- audit table showing sample raw text vs processed text vs compound score
- summary statistics before and after lexicon extension

During EDA, explicitly inspect:

1. sample posts that changed materially after preprocessing/lexicon extension
2. posts containing overridden self-identifier terms
3. posts containing squeeze/calls/puts/yolo language
4. distribution shift in sentiment scores before vs after WSB adaptation

The point is to verify that the custom lexicon improves domain fit without injecting opaque rules.

### Event-level sentiment features

Use exactly these six event-level sentiment features:

1. `post_count`
2. `mean_sentiment`
3. `median_sentiment`
4. `std_sentiment`
5. `frac_positive`
6. `frac_negative`

Do **not** include in the main model:

- max sentiment
- min sentiment
- weighted sentiment

#### Post-level sentiment thresholds

Using VADER compound score `s`:

- positive if `s > 0.05`
- negative if `s < -0.05`
- neutral otherwise

#### Event-level formulas

Let `N` be matched post count for a given stock-event window.

- `post_count = N`
- `mean_sentiment = mean(compound_score)`
- `median_sentiment = median(compound_score)`
- `std_sentiment = sample std(compound_score)`
- `frac_positive = positive_posts / N`
- `frac_negative = negative_posts / N`

If `N == 0`, do **not** silently drop the event. Record the event and surface the coverage issue in EDA.

---

### Baseline market features

Use these baseline features:

1. `ret5_pre`
2. `ret20_pre`
3. `vol20_pre`
4. `abvol_pre`
5. ticker fixed effects
6. calendar-year fixed effects

Definitions:

- `ret5_pre`: 5-trading-day return ending on the last trading day before the earnings date
- `ret20_pre`: 20-trading-day return ending on the last trading day before the earnings date
- `vol20_pre`: 20-trading-day realized volatility of daily returns prior to the event
- `abvol_pre`: day-minus-1 trading volume divided by trailing 20-trading-day average volume

Do not add extra market features unless explicitly approved.

---

### Models

Build exactly these primary models:

1. **OLS baseline**
2. **OLS baseline + sentiment**
3. **Ridge baseline**
4. **Ridge baseline + sentiment**

No tree models. No neural nets. No LLM classifiers.

---

### Fixed effects

Include:

- **ticker fixed effects**
- **calendar-year fixed effects**

Implementation guidance:

- use categorical encoding / dummies in statsmodels for OLS
- use one-hot encoding for sklearn pipelines where needed
- always omit one reference category to avoid perfect multicollinearity when manually constructing dummies

---

### Train / validation / test design

Do **not** do random train-test splits.

Use time-respecting evaluation.

#### OLS

- fit on training window
- evaluate on final holdout test window

#### Ridge

Use walk-forward / expanding-window validation **inside 2016-2021** to tune alpha.

Recommended pattern:

- train 2016-2018, validate 2019
- train 2016-2019, validate 2020
- train 2016-2020, validate 2021

Then:

- refit Ridge on 2016-2021 using chosen alpha
- evaluate once on 2022-2023 test set

Final holdout test window:

- **2022 through 2023-03-28**

---

### Evaluation metrics

Report these for all model variants:

1. `RMSE`
2. `MAE`
3. `directional_accuracy`

Also report the comparison between:

- baseline model
- baseline + sentiment model

Primary question:

- does adding Reddit sentiment improve out-of-sample performance?

---

## Required EDA before final modeling

Before treating the stock universe as final, generate EDA tables/figures for:

1. number of usable earnings events per stock
2. number of matched Reddit posts per stock-event window
3. distribution of `post_count` by stock
4. share of stock-events with `post_count < 10`
5. share of multi-ticker posts by stock
6. sample examples of matched posts for each ticker
7. false-positive audit for ambiguous tickers
8. sentiment score distribution before and after WSB normalization

The point of this EDA is to validate the universe and matching rules.

Do not silently alter the universe based on EDA. Summarize findings for the user first.

---

## Repository structure (current)

This is the actual layout of the repo. Scripts were numbered/consolidated during implementation; some helpers in the original recommended structure were folded into their calling scripts rather than broken out into standalone utils modules.

```text
nlp/
├── CLAUDE.md
├── DECISIONS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── universe.yaml
│   ├── paths.yaml
│   ├── sentiment.yaml
│   ├── wsb_vader_lexicon.json
│   ├── wsb_phrase_map.json
│   └── wsb_emoji_map.json
├── data/
│   ├── raw/
│   │   ├── reddit/
│   │   ├── wrds/
│   │   └── market/
│   ├── interim/
│   └── processed/
├── utils/
│   ├── __init__.py
│   ├── paths.py              # config loaders
│   ├── wrds_io.py            # WRDS Compustat + CRSP pulls
│   ├── stock_aliases.py      # universe config accessors
│   ├── stock_matcher.py      # cashtag/alias/ticker + ambiguous-cue logic
│   ├── wsb_preprocessing.py  # clean_text, phrase/emoji mapping
│   └── sentiment_scoring.py  # VADER analyzer builder + scoring helpers
├── scripts/
│   ├── 01_load_or_pull_wrds_earnings.py
│   ├── 02_prepare_reddit_posts.py
│   ├── 03_match_posts_to_stocks.py
│   ├── 04_link_posts_to_earnings_windows.py
│   ├── 05_score_sentiment.py
│   ├── 06_aggregate_event_sentiment.py
│   ├── 07_build_market_features.py
│   ├── 08_build_event_dataset.py
│   ├── 09_eda_stock_event_coverage.py
│   ├── 10_train_models.py
│   └── _*.py                 # one-off EDA / conversion scripts
├── notebooks/
│   └── 02_results_overview.ipynb   # presentation of model results
├── outputs/
│   ├── figures/
│   └── tables/
└── tests/
```

Notes:

- `utils/` is for reusable logic.
- `scripts/` is for one-off pipeline stages (numbered in execution order).
- Scripts prefixed with `_` are one-off EDA / probes kept for audit, not part of the main pipeline.
- `data/raw/` should be gitignored.
- Processed datasets and final modeling tables are reproducible from raw inputs by running scripts 01–10 in order.
- Helpers that the original plan scoped as standalone utils (`market_features.py`, `event_dataset.py`, `modeling.py`, `evaluation.py`) ended up inlined in their corresponding numbered scripts, which is in line with the spec's "one-off execution scripts in scripts/" rule.

---

## Required file responsibilities

### `configs/universe.yaml`

Store the locked stock universe, canonical company names, and high-level alias metadata.

### `configs/sentiment.yaml`

Store sentiment pipeline options such as:

- whether to use custom lexicon extension
- positive/negative VADER thresholds
- whether to emit audit tables
- whether to keep auxiliary neutral jargon features later

### `configs/wsb_vader_lexicon.json`

Store the custom WSB token-to-score mapping used to update the VADER lexicon.

### `configs/wsb_phrase_map.json`

Store deterministic phrase replacements such as:

- `to the moon` -> `to_the_moon`
- `diamond hands` -> `diamond_hands`
- `hold the line` -> `hold_the_line`

### `configs/wsb_emoji_map.json`

Store emoji-to-token replacements used in preprocessing.

### `utils/stock_matcher.py`

Implements post-to-stock matching with audit-friendly regex logic and ambiguous-ticker safeguards.

### `utils/wsb_preprocessing.py`

Implements deterministic phrase replacement and emoji mapping for sentiment preparation.
This module should not make fuzzy or probabilistic replacements.

### `utils/sentiment_scoring.py`

Initializes VADER, applies the custom lexicon update, scores processed text, and returns post-level sentiment outputs.
It should also support comparison mode:

- raw VADER on minimally cleaned text
- adapted VADER on WSB-preprocessed text

### Event-dataset assembly

Merging earnings metadata, market features, and aggregated Reddit sentiment into the final event-level modeling table is done directly in `scripts/08_build_event_dataset.py`. The logic is small enough that it did not need a standalone `utils/` module.

### `utils/wrds_io.py`

Handles WRDS access: Compustat `fundq` for earnings report dates (`rdq`), and CRSP `dsf` joined with `dsenames` for daily prices/returns/volume. Supports both direct WRDS pulls and loading cached parquet/CSV from `data/raw/`.

---

## Implementation priorities

Build the pipeline in this order:

1. validate raw inputs and paths
2. load WRDS earnings dates
3. prepare Reddit posts
4. match posts to stocks
5. link matched posts to earnings windows
6. run WSB phrase/emoji preprocessing
7. score posts with raw and adapted VADER
8. aggregate event-level sentiment features
9. build market features
10. build final event-level dataset
11. run coverage and sentiment EDA
12. fit OLS and Ridge models
13. produce final tables and figures

---

## Things the agent must not do

- do not silently rewrite the stock universe
- do not replace the custom lexicon with a black-box slang translator
- do not treat `calls`, `puts`, `dd`, or `yolo` as automatically directional sentiment without approval
- do not let sentiment preprocessing change stock matching behavior
- do not use comments in v1
- do not add unapproved features or models
- do not optimize away auditability

---

## Deliverables the repo should produce

At minimum, the pipeline should generate:

1. cleaned earnings-event table
2. matched Reddit-post table with matched tickers and event links
3. processed-text audit table
4. event-level sentiment feature table
5. final modeling dataset
6. EDA tables and figures
7. OLS and Ridge performance comparison tables
8. writeup-ready figures and summary outputs
