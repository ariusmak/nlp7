# Decision Log

Concise log of the dataset and universe decisions made so far, and why. Newest at the bottom within each section.

---

## Reddit data source

**Decision:** Use `p4p_reddit_posts.parquet` (2,275,310 posts) instead of the original `r_wallstreetbets_posts.csv`.

**Why:** The p4p dump includes post **body text** (`selftext`), not just titles, and has ~2× more posts. More signal for VADER and more reliable ticker matching.

**Coverage window of the source:** 2012-04-11 → 2023-03-28.

---

## Rejected: combined legacy CSV + 2022 dump (7-stock setup)

**Decision:** Did **not** use the combined `posts.csv + wallstreetbets_2022.csv` on a 7-ticker shortlist (AAPL, GME, MCD, MSFT, NFLX, NVDA, TSLA).

**Why the experiment happened:** the legacy CSV was pre-filtered to those 7 tickers. We ran `scripts/_eda_7stock_combined.py` twice — once across all subreddits, once filtered to `subreddit == wallstreetbets`.

**Why we rejected it:**

| version                     | posts     | real-body % | matched % | top-ticker share | peak-year share |
|-----------------------------|-----------|-------------|-----------|------------------|-----------------|
| all subreddits, combined    | 778,160   | 35.7%       | 94.6%     | —                | —               |
| WSB-only subset             | 284,121   | **18.1%**   | 85.2%     | **GME 83.3%**    | **2021 = 82%**  |

Under the WSB-only filter (the only sample scope we'd actually use), GME absorbed 83% of matched posts and 2021 alone held 82% of the year-volume. Ticker and year FE couldn't rescue that — the signal would be dominated by one stock in one meme year. MCD also fell to 703 posts across 7 years, well below the event-level threshold. The p4p path was strictly better.

---

## Rejected: 7-stock universe

**Decision:** Not adopted. 7-stock shortlist (AAPL, GME, MCD, MSFT, NFLX, NVDA, TSLA) was an artifact of the legacy CSV's pre-filtering, not a deliberate selection.

**Why we expanded back to 15:** the combined-dataset experiment above killed the 7-stock path. Once we committed to the p4p source, we had to pick a universe from scratch. GME was excluded explicitly (meme-spike distorts time series — the user's rule was "consistently mentioned across years, not 1 spike"). MCD was too thin. That left 5 usable tickers, below a workable minimum. We rebuilt to 15 tickers using the p4p match volumes as the gating criterion (≥1,000 matched posts in the safe window).

---

## Sample window

**Decision:** Restrict earnings events to **2016-01-01 → 2023-03-28**.

**Why:**
- **End (2023-03-28):** last day of the Reddit dump.
- **Start (2016-01-01):** pre-2016 WSB was too small. Early EDA showed ~79% of 2012–2014 earnings events had **zero** matched posts in their pre-earnings windows. Including them would add noise, not signal.

**Impact:** 435 earnings events across the 15 tickers, 97.7% with ≥1 matched post.

---

## Stock universe

**Decision:** Locked 15-ticker universe:
AAPL, AMD, AMZN, BA, DIS, GOOGL, GS, INTC, JPM, MSFT, MU, NFLX, NVDA, TSLA, WMT.

Ambiguous-flagged (stricter matching required): AMD, BA, DIS, GS, MU.

**Why (swaps from the original draft list):**
- Original list included BAC, MCD, QCOM, NKE. Coverage EDA showed these had too few matched posts to support event-level sentiment aggregation.
- Replacements had to satisfy two criteria: (1) **≥1,000 matched posts** under the locked window, (2) **consistent mentions across years**, not one meme-spike (so GME, BB, NOK were explicitly excluded — they distort time-series sentiment).
- Single-letter / highly ambiguous tickers (e.g., `F` for Ford, `T` for AT&T) were rejected — too hard to match cleanly without heavy false positives. `COST` was skipped (common English word). `BABA` fails the full-window rule (IPO 2014). `PFE` carries a 2020–21 vaccine spike.
- **NKE was attempted and dropped.** NKE's in-window match volume was below the ~1,000 comfort threshold. When we probed financials sector replacements, GS hit 1,592 in-window matches (935 with real body) — the best candidate — and restored bank-sector coverage alongside JPM that BAC had provided.
- Final additions: **BA, MU, GS** (replacing BAC, MCD, QCOM, and interim NKE).

**Result:** every ticker clears the 1,000-match floor. Global match counts (stage 03): WMT is the floor at 1,738; TSLA the ceiling at ~38k. Per-ticker counts restricted to each ticker's own 10-day pre-earnings windows (stage 04): GS floor at 339 links; TSLA ceiling at 5,997 links.

---

## Reddit event window (pre-earnings lookback)

**Decision:** **10 calendar days** before the earnings date, excluding the earnings date itself.

**Why:** Originally 7 days. Coverage EDA on the 15-ticker universe showed low-volume tickers (GS, BA, WMT, INTC, JPM) had thin-coverage rates (<10 posts per event) that hurt event-level aggregation — `std_sentiment` and `frac_positive/negative` are unstable when `N` is small. Widening to 10 days preserves the pre-earnings focus while pulling more posts into low-volume events.

**Impact (7-day → 10-day, combined with expanded matching):**

| Metric                  | 7-day, 15 aliases | 10-day, expanded |
|-------------------------|-------------------|------------------|
| Events with ≥1 post     | 96.6%             | **97.7%**        |
| Zero-post events        | 15                | **10**           |
| Thin events (<10 posts) | 33.8%             | **26.0%**        |
| Post-event links        | 13,965            | **20,700**       |
| Real-body links         | 6,443             | **8,833**        |

---

## Post-to-stock matching

**Decision:** Tiered regex match — cashtag (`$TSLA`) → exact ticker token (`TSLA`) → company-name alias (`Tesla`). Word-boundary regex, case-insensitive. Multi-stock posts are assigned to all matched tickers and flagged.

**Ambiguous tickers** (AMD, BA, DIS, GS, MU) require **cashtag OR alias OR ticker + nearby finance cue**. Bare ticker alone is not enough — too many false positives (e.g., "BA" in plain prose, "GS" as an abbreviation).

**Why:** Naive substring matching produces obvious false positives; word boundaries + the ambiguous-ticker safeguard keep the match table auditable.

**Recent refinement — expanded aliases and finance cues:**
- **Aliases:** added product and executive names (e.g., `iPhone`, `Azure`, `Ryzen`, `Cybertruck`, `Elon Musk`, `Jensen Huang`, `Jamie Dimon`). WSB posts frequently reference products/CEOs without naming the ticker.
- **Finance cues:** extended from the 10 terms locked in CLAUDE.md to 22, adding `options`, `dividend`, `revenue`, `analyst`, `upgrade`, `downgrade`, `price target`, `position`, `quarter`, `report`, `beats`, `miss`, `YOLO`. (CLAUDE.md allows for auditable cue expansion; the full list lives in `configs/universe.yaml`.)
- This change, combined with the 10-day window, drove the coverage gains in the table above.

**Regex-boundary choice (auditable):** the matcher uses `(?<![A-Za-z0-9_])TOKEN(?![A-Za-z0-9_])` rather than `\b`. Reason: `\b` treats `$` as a word boundary, which would break cashtag matching (`$TSLA` would match the bare-ticker regex for `TSLA` on the right side but mis-tokenize on the left). Custom boundary keeps cashtags and bare tickers distinctly classifiable. All matching is case-insensitive.

**Stage 03 vs Stage 04 note (for groupmates reading per-ticker counts).** Stage 03 (`03_match_posts_to_stocks.py`) emits the universe of posts matched to any ticker *regardless of window* (104,018 post-ticker rows). Stage 04 (`04_link_posts_to_earnings_windows.py`) applies the per-ticker 10-day pre-earnings window → 20,700 post-event links. Do not compare raw stage-03 counts to event-level sentiment — only stage-04 links enter the model.

---

## Ambiguous-ticker floor (`GS`)

**Decision:** Accept GS as the lowest-coverage ticker (mean ~11.7 posts/event, 13 thin events).

**Why:** It was the best candidate in the replacement sweep (1,592 in-window matches, 935 with real bodies — highest among tested financial-sector replacements). Further tightening the universe would drop a sector (financials ex-JPM) we want represented. Trade-off accepted.

---

## Target / baseline design (locked but unchanged)

These were locked in the initial project scope and have **not** been revisited:

- **Target:** 3-trading-day post-earnings return, with `t0` selected by earnings timing (before open → earnings day close; after close → next day close; missing → previous day close fallback with a flag).
- **Baseline features:** `ret5_pre`, `ret20_pre`, `vol20_pre`, `abvol_pre`, ticker FE, year FE.
- **Models:** OLS and Ridge, each with/without sentiment.
- **Train/test:** walk-forward validation within 2016–2021; holdout = 2022 through 2023-03-28.

**Timing-branch note.** WRDS `comp.fundq` (the earnings source) does **not** carry before-open / after-close metadata. All 435 events land in the `fallback_prev_close` branch of the timing rule. The `earnings_timing` column is preserved as `"unknown"` for auditability; the `timing_used` flag is uniformly `"fallback_prev_close"`. A future extension using IBES or earnings-call-transcript timing would populate the before/after branches for real.

---

## Sentiment methodology (locked)

**Decision:** VADER with a custom WSB lexicon extension + deterministic phrase/emoji preprocessing. **Not** FinBERT as the primary method; not free-form "translation."

**Why:** The mechanism must be transparent and inspectable for the writeup. `analyzer.lexicon.update(custom_wsb_lexicon)` teaches VADER domain vocabulary without obscuring what changed. FinBERT is reserved as an optional robustness check, not the main model.

**Neutral overrides that matter:** `retard`, `autist`, `smooth_brain` → 0.0 (WSB self-deprecation, not negative sentiment). `calls`, `puts`, `dd`, `yolo` → 0.0 (trading style, not directional sentiment).

---

## Modeling outcome (2022–2023 holdout, N = 75)

| variant          | test RMSE | test MAE | dir. acc. | α (CV) |
|------------------|-----------|----------|-----------|--------|
| OLS baseline     | 0.0949    | 0.0687   | 57.3%     | —      |
| OLS + sentiment  | 0.0936    | 0.0677   | 54.7%     | —      |
| Ridge baseline   | 0.0948    | 0.0701   | 53.3%     | 1e5    |
| Ridge + sentiment| 0.0948    | 0.0701   | 53.3%     | ~5.6e4 |

**Null result.** Adding WSB sentiment to the baseline gives a tiny RMSE/MAE improvement in OLS (−1.4% RMSE) but *lowers* directional accuracy by 2.7 pp. Walk-forward Ridge selects α≈10⁵ — effectively intercept-only — under both feature sets, and the two Ridge variants are numerically indistinguishable on the holdout. No sentiment coefficient is individually significant at the 10% level in-sample. This is the headline of the writeup, not a flaw.

---

## QA-identified items (noted here so groupmates know what's known)

- **Target returns use unadjusted CRSP `prc` ratio** (`scripts/07`), while pre-event returns use the adjusted `ret` series. A split inside the 3-day target window would give a wildly wrong target. **Verified: no earnings event in our 435-event sample has a split inside its 3-day post-window** (AAPL, TSLA, NVDA, AMZN, GOOGL all miss). Flagged as a latent bug worth fixing before any universe/window change.
- **Year fixed effects for 2022 and 2023 are exactly 0** in all four models, because those years do not exist in training. This is expected under the spec, but means year FE cannot extrapolate onto the holdout — the test events are implicitly predicted with the 2016 year-FE baseline.
- **UTC vs ET boundary**: Reddit timestamps are UTC; earnings `rdq` is a naive date. Posts from the last ~4 hours of the evening before earnings (ET) normalize to the earnings day in UTC and get excluded. Small bias vs the spec's "exclude earnings-day itself" rule; documented here so it's not a surprise.
- **"Raw VADER" in stage 05 audit tables runs on cleaned (URL/markdown-stripped) text, not truly raw.** Fair A/B for the WSB adaptation layer; just call it out accurately in the writeup ("VADER on cleaned text" vs "VADER on WSB-preprocessed text + extended lexicon").
