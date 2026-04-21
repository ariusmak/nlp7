from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from utils.paths import paths


def load_tm():
    spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "10_train_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm


def fit_variant(tm, df: pd.DataFrame, label: str, sentiment_feats: list[str] | None) -> dict:
    tm.SENTIMENT_FEATS = sentiment_feats if sentiment_feats is not None else []
    df = df.sort_values(["rdq", "ticker"]).reset_index(drop=True)
    train_mask = df["calendar_year"].isin(range(2016, 2022)).values
    test_mask = df["calendar_year"].isin([2022, 2023]).values
    include_sent = bool(sentiment_feats)
    r = tm.run_variant(df, "OLS", include_sent, train_mask, test_mask)
    ols = r["ols_result"]
    coefs = pd.DataFrame({
        "feature": ols.params.index,
        "coef": ols.params.values,
        "std_err": ols.bse.values,
        "tvalue": ols.tvalues.values,
        "pvalue": ols.pvalues.values,
    })
    coefs["variant"] = label
    summary = {
        "variant": label,
        "n_train": int(r["train"]["n"]),
        "n_test": int(r["test"]["n"]),
        "n_params": int(len(ols.params)),
        "r2": float(ols.rsquared),
        "r2_adj": float(ols.rsquared_adj),
        "f_stat": float(ols.fvalue),
        "f_pvalue": float(ols.f_pvalue),
        "train_RMSE": r["train"]["RMSE"],
        "train_MAE":  r["train"]["MAE"],
        "train_dir_acc": r["train"]["dir_acc"],
        "test_RMSE": r["test"]["RMSE"],
        "test_MAE":  r["test"]["MAE"],
        "test_dir_acc": r["test"]["dir_acc"],
    }
    return {"coefs": coefs, "summary": summary}


def star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    processed = p["data"]["processed"]
    tables = p["outputs"]["tables"]
    tables.mkdir(parents=True, exist_ok=True)

    tm = load_tm()
    vader_df = pd.read_parquet(processed / "event_modeling_dataset.parquet")
    finbert_df = pd.read_parquet(processed / "event_modeling_dataset_finbert.parquet")

    sent_feats = ["post_count", "mean_sentiment", "median_sentiment",
                  "std_sentiment", "frac_positive", "frac_negative"]

    res_baseline = fit_variant(tm, vader_df, "ols_baseline", None)
    res_vader    = fit_variant(tm, vader_df, "ols_vader", sent_feats)
    res_finbert  = fit_variant(tm, finbert_df, "ols_finbert", sent_feats)

    all_coefs = pd.concat([res_baseline["coefs"], res_vader["coefs"], res_finbert["coefs"]],
                          ignore_index=True)
    all_coefs["sig"] = all_coefs["pvalue"].apply(star)
    all_coefs_out = tables / "final_comparison_coefficients_long.csv"
    all_coefs.to_csv(all_coefs_out, index=False)

    wide = all_coefs.pivot_table(
        index="feature", columns="variant",
        values=["coef", "pvalue"], aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    feat_order = (
        ["const", "ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre"]
        + sent_feats
        + [c for c in wide.index if c.startswith("tk_")]
        + [c for c in wide.index if c.startswith("yr_")]
    )
    feat_order = [f for f in feat_order if f in wide.index] + \
                 [f for f in wide.index if f not in feat_order]
    wide = wide.reindex(feat_order)
    wide_out = tables / "final_comparison_coefficients_wide.csv"
    wide.to_csv(wide_out)

    summary_df = pd.DataFrame([res_baseline["summary"], res_vader["summary"], res_finbert["summary"]])
    summary_out = tables / "final_comparison_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    print("=" * 78)
    print("SUMMARY STATS")
    print("=" * 78)
    cols_show = ["variant", "n_train", "n_test", "n_params",
                 "r2", "r2_adj", "f_stat", "f_pvalue",
                 "train_RMSE", "test_RMSE", "test_MAE", "test_dir_acc"]
    print(summary_df[cols_show].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    ))

    print()
    print("=" * 78)
    print("CORE FEATURE COEFFICIENTS (baseline + sentiment)  [coef  (p-value) sig]")
    print("=" * 78)
    core = ["const", "ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre"] + sent_feats
    hdr = f"{'feature':<20} {'baseline':>20} {'vader':>20} {'finbert':>20}"
    print(hdr)
    print("-" * len(hdr))
    for f in core:
        if f not in wide.index:
            continue
        def cell(col_prefix):
            c = wide.loc[f, f"coef_{col_prefix}"] if f"coef_{col_prefix}" in wide.columns else np.nan
            pv = wide.loc[f, f"pvalue_{col_prefix}"] if f"pvalue_{col_prefix}" in wide.columns else np.nan
            if pd.isna(c):
                return "     —"
            s = star(pv)
            return f"{c:+.5f} ({pv:.3f}){s}"
        print(f"{f:<20} {cell('ols_baseline'):>20} {cell('ols_vader'):>20} {cell('ols_finbert'):>20}")

    print()
    print("=" * 78)
    print("TICKER FIXED EFFECTS  [coef  (p-value)]")
    print("=" * 78)
    tk_feats = [c for c in wide.index if c.startswith("tk_")]
    for f in tk_feats:
        def cell(col_prefix):
            c = wide.loc[f, f"coef_{col_prefix}"] if f"coef_{col_prefix}" in wide.columns else np.nan
            pv = wide.loc[f, f"pvalue_{col_prefix}"] if f"pvalue_{col_prefix}" in wide.columns else np.nan
            if pd.isna(c):
                return "     —"
            return f"{c:+.4f} ({pv:.3f}){star(pv)}"
        print(f"{f:<20} {cell('ols_baseline'):>20} {cell('ols_vader'):>20} {cell('ols_finbert'):>20}")

    print()
    print("=" * 78)
    print("YEAR FIXED EFFECTS  [coef  (p-value)]")
    print("=" * 78)
    yr_feats = [c for c in wide.index if c.startswith("yr_")]
    for f in yr_feats:
        def cell(col_prefix):
            c = wide.loc[f, f"coef_{col_prefix}"] if f"coef_{col_prefix}" in wide.columns else np.nan
            pv = wide.loc[f, f"pvalue_{col_prefix}"] if f"pvalue_{col_prefix}" in wide.columns else np.nan
            if pd.isna(c):
                return "     —"
            return f"{c:+.4f} ({pv:.3f}){star(pv)}"
        print(f"{f:<20} {cell('ols_baseline'):>20} {cell('ols_vader'):>20} {cell('ols_finbert'):>20}")

    print()
    print("significance: * p<0.10  ** p<0.05  *** p<0.01")
    print()
    print(f"wrote: {all_coefs_out}")
    print(f"wrote: {wide_out}")
    print(f"wrote: {summary_out}")


if __name__ == "__main__":
    main()
