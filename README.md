# nlp

## Project overview

This project asks a simple question:

**Does WallStreetBets Reddit sentiment improve prediction of short-horizon post-earnings stock returns?**

The goal is to compare two models:

1. a **baseline model** built from standard pre-earnings market features
2. a **sentiment-augmented model** that adds Reddit sentiment derived from WallStreetBets posts

The project is intentionally designed to be clean and explainable for a class setting. We are not trying to build a production trading system. We are trying to test whether Reddit text adds measurable predictive value around earnings events.

---

## Core research question

For a fixed stock universe, does adding WallStreetBets sentiment to a standard set of pre-earnings features improve prediction of the stock's **3-trading-day return after earnings**?

---

## High-level approach

We build the dataset at the **earnings-event level**.

That means:

- each row represents **one stock and one earnings announcement**
- for each event, we collect:
  - standard market features from before earnings
  - Reddit posts from the 10 calendar days before earnings
  - sentiment features derived from those posts
  - the stock's 3-trading-day return after the earnings event

We then compare model performance with and without the Reddit features.

---

## Data sources

### 1. Reddit data

Primary text data comes from a WallStreetBets Reddit dump (`p4p_reddit_posts.parquet`, 2,275,310 posts with title + selftext) covering:

- **Earliest timestamp:** `2012-04-11 16:40:40 UTC`
- **Latest timestamp:** `2023-03-28 22:11:08 UTC`

This dataset is used to identify pre-earnings discussion and compute sentiment features.

### 2. Earnings dates

Historical earnings announcement dates (`rdq`) are pulled from **WRDS Compustat** (`comp.fundq`). These dates define the event windows used throughout the project. Note: `fundq` does not carry before-open / after-close timing metadata, so all 435 events fall into the `fallback_prev_close` branch of the timing rule below.

### 3. Market data

Daily price and volume data come from **WRDS CRSP** (`crsp.dsf` joined with `crsp.dsenames`). They are used to construct:

- the post-earnings return target
- standard pre-earnings baseline features

---

## Sample window

Because the Reddit dataset begins and ends at fixed timestamps, the usable earnings-event window is narrower than the raw Reddit coverage window.

We use a **safe event window** of:

- **Start:** `2016-01-01`
- **End:** `2023-03-28`

The start was narrowed from the earliest Reddit date (2012-04-11) to 2016-01-01 because WSB was too small pre-2016 (most 2012-2014 earnings events had zero matched posts in their pre-earnings windows). Under the current window (10-day pre-earnings lookback, expanded alias + finance-cue matching), 435 earnings events cover all 15 tickers with 97.7% of events having ≥1 matched post.

---

## Locked stock universe

The locked 15-stock universe is:

- AAPL — Apple
- AMD — Advanced Micro Devices *(ambiguous)*
- AMZN — Amazon
- BA — Boeing *(ambiguous)*
- DIS — Disney *(ambiguous)*
- GOOGL — Alphabet / Google
- GS — Goldman Sachs *(ambiguous)*
- INTC — Intel
- JPM — JPMorgan Chase
- MSFT — Microsoft
- MU — Micron *(ambiguous)*
- NFLX — Netflix
- NVDA — NVIDIA
- TSLA — Tesla
- WMT — Walmart

Ambiguous-flagged tickers require cashtag, company-name alias, or ticker-plus-finance-cue matches to count.

Under the locked safe window and p4p source, every ticker clears ~1,700 globally-matched posts (WMT is the floor at 1,738; TSLA the ceiling at ~38k). After restricting to each ticker's own 10-day pre-earnings windows, per-ticker post-event links range from 339 (GS) to 5,997 (TSLA) across 435 events.

---

## Unit of analysis

The unit of analysis is:

**one stock-earnings event**

For each event, we combine:

- earnings metadata
- market features
- event-window Reddit sentiment features
- the post-earnings return target

---

## Target variable

The target is the stock's **3-trading-day post-earnings return**.

If:

- earnings are announced **before market open**, the return window starts from the **earnings-day close**
- earnings are announced **after market close**, the return window starts from the **next trading day close**
- if timing is missing, we use the **previous trading day close** as the fallback starting price

This rule is meant to align the return window with when the market can incorporate the earnings information.

---

## Reddit event window

For each earnings event, we collect WallStreetBets posts from the:

**10 calendar days before the earnings date**, excluding the earnings date itself.

We do not use post timestamps within the earnings day itself in the main specification. The window was widened from 7 to 10 days after coverage EDA showed several low-volume tickers (GS, BA, WMT) had thin-coverage rates that degraded event-level sentiment aggregation.

---

## How posts are linked to stocks

A Reddit post is linked to a stock using a tiered matching rule:

1. **cashtags**, such as `$TSLA`
2. **exact ticker tokens**, such as `TSLA`
3. **company-name aliases**, such as `Tesla`, `Apple`, `Google`, or `Nvidia`

This matching is done carefully to avoid false positives. For more ambiguous ticker strings, stricter context checks are used.

Posts that mention more than one stock are assigned to all matched stocks and flagged as multi-ticker posts.

---

## Sentiment methodology

The main sentiment pipeline is:

1. clean text
2. preprocess selected WallStreetBets phrases and emojis into stable tokens
3. extend VADER with a **custom WallStreetBets lexicon**
4. score the processed text using **VADER compound sentiment**

This is an important design choice. Rather than trying to "translate" Reddit language into standard English, we adapt VADER directly to the WallStreetBets domain. In practice, that means teaching VADER how to score community-specific slang.

Examples of phrase preprocessing include:

- `to the moon` → `to_the_moon`
- `diamond hands` → `diamond_hands`
- `hold the line` → `hold_the_line`
- `short squeeze` → `short_squeeze`

Examples of emoji preprocessing include:

- `🚀` → `rocket`
- `💎` → `diamond`
- `💎🙌` → `diamond_hands`
- `🦍` → `ape`

Then those tokens can be given explicit sentiment values in a custom VADER lexicon.

Examples of useful WallStreetBets sentiment tokens include:

- `to_the_moon`
- `rocket`
- `tendies`
- `diamond_hands`
- `paper_hands`
- `bagholder`
- `rug_pull`
- `dead_cat_bounce`

We also override some terms that VADER would normally interpret too negatively in a general-language setting, but that often function as self-deprecating or community-neutral language on WallStreetBets (e.g. `retard`, `autist`, `smooth_brain` → 0.0).

The whole sentiment adaptation layer is intentionally explicit and inspectable so it can be described clearly in the writeup.

---

## Event-level sentiment features

For each stock-earnings event, we aggregate post-level sentiment into six features:

1. `post_count`
2. `mean_sentiment`
3. `median_sentiment`
4. `std_sentiment`
5. `frac_positive`
6. `frac_negative`

Using VADER's standard compound thresholds:

- positive if `compound > 0.05`
- negative if `compound < -0.05`
- neutral otherwise

These features summarize both the amount of discussion and the tone of discussion in the pre-earnings window.

We intentionally do **not** use max sentiment, min sentiment, or upvote-weighted sentiment in the main specification. We also do not automatically treat words such as `calls`, `puts`, `dd`, or `yolo` as directional sentiment in the primary model because those terms often describe trading style or instrument choice rather than clear positive or negative opinion.

---

## Baseline market features

The baseline model uses standard pre-earnings features:

- **5-day pre-earnings return**
- **20-day pre-earnings return**
- **20-day realized volatility**
- **abnormal volume** relative to the trailing 20-day average
- **ticker fixed effects**
- **calendar-year fixed effects**

These features create a simple, interpretable benchmark before adding any NLP information.

---

## Models

We estimate two model families:

### 1. OLS

- baseline only
- baseline + Reddit sentiment

### 2. Ridge regression

- baseline only
- baseline + Reddit sentiment

This keeps the modeling framework simple and appropriate for a relatively small event-level dataset.

---

## Train / test design

Because this is time-series event data, we do **not** use random train-test splits.

- Ridge α is tuned by **walk-forward validation inside 2016–2021** (folds: train 2016–18 → val 2019; train 2016–19 → val 2020; train 2016–20 → val 2021).
- The holdout test window is **2022 through 2023-03-28** (75 events).
- Training set is 2016–2021 (360 events).

This preserves the time ordering of information and avoids leakage.

---

## Evaluation metrics

We compare models on the holdout using:

- **RMSE**
- **MAE**
- **directional accuracy**

The main question is whether the sentiment-augmented model performs better out of sample than the baseline model. See [notebooks/02_results_overview.ipynb](notebooks/02_results_overview.ipynb) for the current outcome — short version: sentiment does not reliably improve out-of-sample performance in this setup.

---

## EDA

Coverage and sentiment EDA is produced by [scripts/09_eda_stock_event_coverage.py](scripts/09_eda_stock_event_coverage.py) and lands in [outputs/figures/](outputs/figures/) and [outputs/tables/](outputs/tables/). It covers:

- number of earnings events per stock
- distribution of matched `post_count` per event, by ticker
- share of events with thin Reddit coverage (<10 posts)
- false-positive audit for ambiguous tickers (AMD, BA, DIS, GS, MU)
- sentiment distributions before vs after the WSB lexicon extension
- univariate correlations between each feature and the target

These outputs validated the 15-ticker universe and the matching rules before modeling.

---

## Why this project is interesting

This project sits at the intersection of:

- event studies
- financial prediction
- sentiment analysis
- domain-specific text preprocessing

A core contribution is not just "use sentiment," but rather:

**use a sentiment pipeline that is adapted to the way Reddit users on WallStreetBets actually communicate.**


Concretely, that adaptation is implemented through phrase preprocessing and a custom VADER lexicon extension, not through an opaque black-box translation step.

That makes the project more meaningful than simply applying an off-the-shelf sentiment model to raw Reddit text.

---

## Outputs

The pipeline produces:

- cleaned earnings-event table ([data/raw/wrds/earnings_dates.parquet](data/raw/wrds/earnings_dates.parquet))
- matched Reddit posts linked to earnings windows ([data/interim/posts_linked_to_events.parquet](data/interim/posts_linked_to_events.parquet))
- event-level sentiment features ([data/processed/event_sentiment_features.parquet](data/processed/event_sentiment_features.parquet))
- final modeling dataset, 435 events ([data/processed/event_modeling_dataset.parquet](data/processed/event_modeling_dataset.parquet))
- EDA figures and tables in [outputs/figures/](outputs/figures/) and [outputs/tables/](outputs/tables/)
- OLS and Ridge comparison metrics ([outputs/tables/model_metrics.csv](outputs/tables/model_metrics.csv))
- results notebook for the writeup ([notebooks/02_results_overview.ipynb](notebooks/02_results_overview.ipynb))

---

## Repository structure

```text
nlp/
├── README.md
├── CLAUDE.md
├── configs/
│   ├── universe.yaml
│   ├── sentiment.yaml
│   ├── wsb_vader_lexicon.json
│   ├── wsb_phrase_map.json
│   └── wsb_emoji_map.json
├── data/
├── utils/
├── scripts/
├── notebooks/
├── outputs/
└── tests/
```

The codebase is structured so that:

- reusable logic lives in `utils/`
- pipeline steps live in `scripts/`
- exploration and presentation live in `notebooks/`
- final figures and tables live in `outputs/`

---

## Practical caveats

A few limitations should be kept in mind:

- some stocks may receive uneven Reddit attention over time
- some posts mention multiple stocks rather than one clear target
- WallStreetBets language is noisy and highly informal
- the event-level sample is still modest compared with large-scale ML datasets

These are not flaws that invalidate the project. They are part of the reason the design is intentionally simple and transparent.

---

## Bottom line

This project is a structured test of whether **domain-adapted Reddit sentiment** adds value beyond standard pre-earnings market features when predicting short-horizon post-earnings returns.

The project is designed to be:

- methodologically clear
- reproducible
- easy to explain in a class writeup
- realistic in scope for a student group project

