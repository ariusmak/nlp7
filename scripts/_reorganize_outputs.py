from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import paths, sentiment_config

POS_THR = 0.05
NEG_THR = -0.05


def load_tm():
    spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "10_train_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm


def try_git_mv(src: Path, dst: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "mv", "-f", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass
    if src.exists():
        shutil.move(str(src), str(dst))
        return True
    return False


def rename_vader_outputs(tables: Path, figs: Path) -> list[tuple[str, str]]:
    renames_tables = [
        ("coefficients_ols_sentiment.csv", "coefficients_ols_vader.csv"),
        ("coefficients_ridge_sentiment.csv", "coefficients_ridge_vader.csv"),
        ("model_metrics.csv", "model_metrics_vader.csv"),
        ("ridge_cv_detail.csv", "ridge_cv_detail_vader.csv"),
        ("event_sentiment_summary.csv", "event_sentiment_summary_vader.csv"),
        ("event_sentiment_per_ticker.csv", "event_sentiment_per_ticker_vader.csv"),
        ("sentiment_audit_top50_movers.csv", "sentiment_audit_top50_movers_vader.csv"),
        ("sentiment_label_confusion.csv", "sentiment_label_confusion_vader.csv"),
        ("sentiment_raw_vs_adapted.csv", "sentiment_raw_vs_adapted_vader.csv"),
        ("probe_mean_sentiment_only_metrics.csv", "probe_mean_sentiment_only_metrics_vader.csv"),
    ]
    renames_figs = [
        ("03_sentiment_hist_raw_vs_adapted.png", "03_sentiment_hist_raw_vs_adapted_vader.png"),
        ("04_event_sentiment_vs_target.png", "04_event_sentiment_vs_target_vader.png"),
        ("06_predictions_vs_actual.png", "06_predictions_vs_actual_vader.png"),
        ("07_residuals_by_variant.png", "07_residuals_by_variant_vader.png"),
    ]
    done = []
    for old, new in renames_tables:
        src, dst = tables / old, tables / new
        if src.exists() and not dst.exists():
            try_git_mv(src, dst)
            done.append((str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))))
    for old, new in renames_figs:
        src, dst = figs / old, figs / new
        if src.exists() and not dst.exists():
            try_git_mv(src, dst)
            done.append((str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))))
    probe_old = tables / "probe_finbert_metrics.csv"
    probe_new = tables / "probe_finbert_metrics.csv"
    if probe_old.exists():
        pass
    return done


def run_finbert_models(tm, df: pd.DataFrame, tables: Path, figs: Path, processed: Path) -> dict:
    df = df.sort_values(["rdq", "ticker"]).reset_index(drop=True)
    train_mask = df["calendar_year"].isin(range(2016, 2022)).values
    test_mask = df["calendar_year"].isin([2022, 2023]).values

    variants = [
        ("OLS", False, "ols_baseline"),
        ("OLS", True, "ols_finbert"),
        ("Ridge", False, "ridge_baseline"),
        ("Ridge", True, "ridge_finbert"),
    ]
    results: dict[str, dict] = {}
    metric_rows = []
    for model_type, include_sent, key in variants:
        r = tm.run_variant(df, model_type, include_sent, train_mask, test_mask)
        results[key] = r
        tr, te = r["train"], r["test"]
        metric_rows.append({
            "variant": key,
            "model": model_type,
            "sentiment": include_sent,
            "alpha": r.get("alpha"),
            "train_RMSE": tr["RMSE"], "train_MAE": tr["MAE"], "train_dir_acc": tr["dir_acc"], "train_n": tr["n"],
            "test_RMSE": te["RMSE"], "test_MAE": te["MAE"], "test_dir_acc": te["dir_acc"], "test_n": te["n"],
        })

    pd.DataFrame(metric_rows).to_csv(tables / "model_metrics_finbert.csv", index=False)

    for key, r in results.items():
        if key in ("ols_baseline", "ridge_baseline"):
            continue
        params = r["params"].copy()
        col = {"coef": params}
        if "pvalues" in r:
            col["pvalue"] = r["pvalues"]
        if "num_scale" in r:
            col["num_scale"] = r["num_scale"]
        coefs = pd.DataFrame(col)
        coefs.index.name = "feature"
        out_key = key
        coefs.to_csv(tables / f"coefficients_{out_key}.csv")

    ridge_cv_rows = []
    for key in ("ridge_baseline", "ridge_finbert"):
        cv = results[key]["cv_detail"].copy()
        cv["variant"] = key
        ridge_cv_rows.append(cv)
    pd.concat(ridge_cv_rows, ignore_index=True).to_csv(tables / "ridge_cv_detail_finbert.csv", index=False)

    preds_df = df.loc[test_mask, ["ticker", "event_id", "rdq", "calendar_year", "target_ret3"]].copy()
    for key, r in results.items():
        preds_df[f"pred_{key}"] = r["predictions_test"]
    preds_df.to_parquet(processed / "model_predictions_test_finbert.parquet", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (_, _, key) in zip(axes.flatten(), variants):
        ax.scatter(preds_df["target_ret3"], preds_df[f"pred_{key}"], s=20, alpha=0.75, color="#2a9d8f")
        lim = max(abs(preds_df["target_ret3"]).max(), abs(preds_df[f"pred_{key}"]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#e76f51", linewidth=1, linestyle="--")
        ax.axhline(0, color="#aaa", linewidth=0.6)
        ax.axvline(0, color="#aaa", linewidth=0.6)
        te = results[key]["test"]
        ax.set_title(f"{key}  (RMSE={te['RMSE']:.4f}, dir_acc={te['dir_acc']:.2f})")
        ax.set_xlabel("actual target_ret3")
        ax.set_ylabel("predicted")
    fig.suptitle("Test-set predictions vs actuals (FinBERT) (2022-2023)")
    fig.tight_layout()
    fig.savefig(figs / "06_predictions_vs_actual_finbert.png", dpi=150)
    plt.close(fig)

    rows = []
    for key in results:
        r = results[key]
        resid = preds_df["target_ret3"].values - r["predictions_test"]
        rows.append(pd.DataFrame({"variant": key, "residual": resid}))
    resid_df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [v[2] for v in variants]
    data = [resid_df.loc[resid_df["variant"] == k, "residual"].values for k in order]
    ax.boxplot(data, showfliers=True, patch_artist=True,
               boxprops=dict(facecolor="#d7ebe7"), medianprops=dict(color="#2a9d8f"))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=0)
    ax.axhline(0, color="#e76f51", linewidth=1)
    ax.set_title("Test-set residual distribution by variant (FinBERT)")
    ax.set_ylabel("residual (actual - predicted)")
    fig.tight_layout()
    fig.savefig(figs / "07_residuals_by_variant_finbert.png", dpi=150)
    plt.close(fig)

    return results


def write_finbert_event_sentiment(df: pd.DataFrame, tables: Path) -> None:
    nonzero = df[df["post_count"] > 0]
    feats = ["post_count", "mean_sentiment", "median_sentiment",
             "std_sentiment", "frac_positive", "frac_negative"]
    summary = pd.DataFrame({
        "feature": feats,
        "mean":   [nonzero[c].mean()   for c in feats],
        "median": [nonzero[c].median() for c in feats],
        "std":    [nonzero[c].std()    for c in feats],
    })
    summary.to_csv(tables / "event_sentiment_summary_finbert.csv", index=False)

    per_ticker = (
        df.groupby("ticker")
          .agg(events=("event_id", "size"),
               zero_events=("post_count", lambda s: int((s == 0).sum())),
               thin_events=("post_count", lambda s: int((s < 10).sum())),
               mean_posts=("post_count", "mean"),
               mean_sent=("mean_sentiment", "mean"),
               mean_fracpos=("frac_positive", "mean"),
               mean_fracneg=("frac_negative", "mean"))
          .sort_values("mean_posts", ascending=False)
    )
    per_ticker["mean_posts"] = per_ticker["mean_posts"].round(1)
    per_ticker["mean_sent"] = per_ticker["mean_sent"].round(4)
    per_ticker["mean_fracpos"] = per_ticker["mean_fracpos"].round(4)
    per_ticker["mean_fracneg"] = per_ticker["mean_fracneg"].round(4)
    per_ticker.to_csv(tables / "event_sentiment_per_ticker_finbert.csv")


def write_finbert_audit(scored: pd.DataFrame, posts_slim: pd.DataFrame, tables: Path) -> None:
    merged = scored.merge(
        posts_slim[["id", "title", "selftext"]], on="id", how="left"
    )
    merged["abs_score"] = merged["fb_score"].abs()
    top = merged.sort_values("abs_score", ascending=False).head(50)

    def _label(s: float) -> str:
        if s > POS_THR:
            return "positive"
        if s < NEG_THR:
            return "negative"
        return "neutral"

    top["fb_label"] = top["fb_score"].apply(_label)
    txt = (top["title"].fillna("").astype(str) + " " + top["selftext"].fillna("").astype(str)).str.slice(0, 400)
    out = pd.DataFrame({
        "ticker": top["ticker"].values,
        "event_id": top["event_id"].values,
        "rdq": top["rdq"].values,
        "post_dt": top["post_dt"].values,
        "fb_p_pos": top["fb_p_pos"].round(4).values,
        "fb_p_neu": top["fb_p_neu"].round(4).values,
        "fb_p_neg": top["fb_p_neg"].round(4).values,
        "fb_score": top["fb_score"].round(4).values,
        "fb_label": top["fb_label"].values,
        "text": txt.values,
    })
    out.to_csv(tables / "sentiment_audit_top50_movers_finbert.csv", index=False)


def write_cross_scorer_confusion(scored_fb: pd.DataFrame, scored_vader: pd.DataFrame, tables: Path) -> None:
    def _label(s: float) -> str:
        if s > POS_THR:
            return "positive"
        if s < NEG_THR:
            return "negative"
        return "neutral"

    fb = scored_fb[["id", "fb_score"]].copy()
    fb["fb_label"] = fb["fb_score"].apply(_label)
    vd = scored_vader[["id", "adp_compound"]].copy()
    vd["vader_label"] = vd["adp_compound"].apply(_label)
    m = fb.merge(vd, on="id", how="inner")
    order = ["negative", "neutral", "positive"]
    cm = pd.crosstab(m["vader_label"], m["fb_label"]).reindex(index=order, columns=order, fill_value=0)
    cm.index.name = "vader_label"
    cm.to_csv(tables / "sentiment_label_confusion_finbert.csv")


def make_finbert_scatter(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["post_count"] >= 10].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    tickers = sorted(sub["ticker"].unique())
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(tickers):
        s = sub[sub["ticker"] == t]
        ax.scatter(s["mean_sentiment"], s["target_ret3"], s=22, alpha=0.7, color=cmap(i % 20), label=t)
    corr = float("nan")
    if len(sub) >= 2:
        x = sub["mean_sentiment"].values
        y = sub["target_ret3"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#264653", linewidth=1.5,
                    linestyle="--", label=f"OLS fit (slope={slope:.3f})")
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.set_title(f"Event mean_sentiment vs 3-day post-earnings return  (FinBERT; N={len(sub)} events >=10 posts; Pearson r = {corr:.3f})")
    ax.set_xlabel("event mean_sentiment (FinBERT)")
    ax.set_ylabel("target_ret3")
    ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_tables_readme(tables: Path) -> None:
    body = """# outputs/tables

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
"""
    (tables / "README.md").write_text(body, encoding="utf-8")


def write_figures_readme(figs: Path) -> None:
    body = """# outputs/figures

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
"""
    (figs / "README.md").write_text(body, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    tables = p["outputs"]["tables"]
    figs = p["outputs"]["figures"]
    processed = p["data"]["processed"]
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    moves = rename_vader_outputs(tables, figs)
    print(f"renamed {len(moves)} files to _vader suffix")
    for src, dst in moves:
        print(f"  {src}  ->  {dst}")

    df_fb = pd.read_parquet(processed / "event_modeling_dataset_finbert.parquet")
    scored_fb = pd.read_parquet(processed / "posts_scored_finbert.parquet")
    scored_vader = pd.read_parquet(processed / "posts_scored.parquet")
    posts_slim = pd.read_parquet(ROOT / "data" / "handoff" / "posts_linked_to_events_slim.parquet")

    print()
    print("fitting OLS + Ridge with FinBERT sentiment...")
    tm = load_tm()
    run_finbert_models(tm, df_fb, tables, figs, processed)
    print("  wrote model_metrics_finbert.csv, coefficients_ols_finbert.csv, "
          "coefficients_ridge_finbert.csv, ridge_cv_detail_finbert.csv")
    print("  wrote 06_predictions_vs_actual_finbert.png, 07_residuals_by_variant_finbert.png")

    write_finbert_event_sentiment(df_fb, tables)
    print("  wrote event_sentiment_summary_finbert.csv, event_sentiment_per_ticker_finbert.csv")

    write_finbert_audit(scored_fb, posts_slim, tables)
    print("  wrote sentiment_audit_top50_movers_finbert.csv")

    write_cross_scorer_confusion(scored_fb, scored_vader, tables)
    print("  wrote sentiment_label_confusion_finbert.csv")

    make_finbert_scatter(df_fb, figs / "04_event_sentiment_vs_target_finbert.png")
    print("  wrote 04_event_sentiment_vs_target_finbert.png")

    write_tables_readme(tables)
    write_figures_readme(figs)
    print("  wrote outputs/tables/README.md and outputs/figures/README.md")


if __name__ == "__main__":
    main()
