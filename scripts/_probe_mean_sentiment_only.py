from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "10_train_models.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

from utils.paths import paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    processed = p["data"]["processed"]

    tm.SENTIMENT_FEATS = ["mean_sentiment"]

    df = pd.read_parquet(processed / "event_modeling_dataset.parquet").reset_index(drop=True)
    df = df.sort_values(["rdq", "ticker"]).reset_index(drop=True)

    train_mask = df["calendar_year"].isin(range(2016, 2022)).values
    test_mask = df["calendar_year"].isin([2022, 2023]).values
    print(f"train events: {int(train_mask.sum())}  (2016-2021)")
    print(f"test  events: {int(test_mask.sum())}   (2022-2023-03-28)")
    print(f"sentiment feature set: {tm.SENTIMENT_FEATS}")
    print()

    variants = [
        ("OLS",   False, "ols_baseline"),
        ("OLS",   True,  "ols_mean_sent"),
        ("Ridge", False, "ridge_baseline"),
        ("Ridge", True,  "ridge_mean_sent"),
    ]

    rows = []
    for model_type, include_sent, key in variants:
        r = tm.run_variant(df, model_type, include_sent, train_mask, test_mask)
        te = r["test"]
        tr = r["train"]
        rows.append({
            "variant": key,
            "model": model_type,
            "sentiment": include_sent,
            "alpha": r.get("alpha"),
            "train_RMSE": tr["RMSE"],
            "train_MAE": tr["MAE"],
            "train_dir_acc": tr["dir_acc"],
            "test_RMSE": te["RMSE"],
            "test_MAE": te["MAE"],
            "test_dir_acc": te["dir_acc"],
            "test_n": te["n"],
        })

        if model_type == "OLS" and include_sent:
            params = r["params"]
            pvals = r["pvalues"]
            if "mean_sentiment" in params.index:
                print(f"  OLS mean_sentiment coef = {params['mean_sentiment']:+.5f}  (p = {pvals['mean_sentiment']:.3f})")
        if model_type == "Ridge" and include_sent:
            params = r["params"]
            if "mean_sentiment" in params.index:
                print(f"  Ridge mean_sentiment coef (scaled) = {params['mean_sentiment']:+.5f}  (alpha = {r['alpha']:.1f})")

    out = pd.DataFrame(rows)
    print()
    print("TEST METRICS:")
    print(out[["variant", "alpha", "test_RMSE", "test_MAE", "test_dir_acc"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    tables_dir = p["outputs"]["tables"]
    out.to_csv(tables_dir / "probe_mean_sentiment_only_metrics.csv", index=False)
    print()
    print(f"wrote: {tables_dir / 'probe_mean_sentiment_only_metrics.csv'}")


if __name__ == "__main__":
    main()
