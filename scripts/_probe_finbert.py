from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.paths import paths

MODEL_ID = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LEN = 512
POS_THRESH = 0.05
NEG_THRESH = -0.05


def load_posts(handoff_dir: Path) -> pd.DataFrame:
    posts = pd.read_parquet(handoff_dir / "posts_linked_to_events_slim.parquet")
    title = posts["title"].fillna("").astype(str)
    body = posts["selftext"].fillna("").astype(str)
    posts["text_for_sentiment"] = (title + " " + body).str.strip()
    return posts


def score_with_finbert(texts: list[str]) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    label_map = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_idx = next(i for i, v in label_map.items() if v == "positive")
    neg_idx = next(i for i, v in label_map.items() if v == "negative")
    neu_idx = next(i for i, v in label_map.items() if v == "neutral")
    print(f"FinBERT label order: {label_map}")

    probs = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="finbert"):
            chunk = texts[start : start + BATCH_SIZE]
            enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")
            logits = model(**enc).logits
            p = softmax(logits, dim=-1).numpy()
            probs[start : start + len(chunk), 0] = p[:, pos_idx]
            probs[start : start + len(chunk), 1] = p[:, neu_idx]
            probs[start : start + len(chunk), 2] = p[:, neg_idx]
    return probs


def aggregate_event_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    pos_label = (scored["fb_score"] > POS_THRESH).astype(int)
    neg_label = (scored["fb_score"] < NEG_THRESH).astype(int)
    scored = scored.assign(_pos=pos_label, _neg=neg_label)

    agg = scored.groupby("event_id").agg(
        post_count=("fb_score", "size"),
        mean_sentiment=("fb_score", "mean"),
        median_sentiment=("fb_score", "median"),
        std_sentiment=("fb_score", "std"),
        frac_positive=("_pos", "mean"),
        frac_negative=("_neg", "mean"),
    ).reset_index()
    return agg


def build_dataset(skeleton: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    df = skeleton.merge(agg, on="event_id", how="left")
    df["post_count"] = df["post_count"].fillna(0).astype(int)
    for c in ["mean_sentiment", "median_sentiment", "std_sentiment",
              "frac_positive", "frac_negative"]:
        df[c] = df[c].fillna(0.0)
    return df


def run_models(df: pd.DataFrame, label: str) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "10_train_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    df = df.sort_values(["rdq", "ticker"]).reset_index(drop=True)
    train_mask = df["calendar_year"].isin(range(2016, 2022)).values
    test_mask = df["calendar_year"].isin([2022, 2023]).values

    rows = []
    for model_type, include_sent, key in [
        ("OLS",   False, f"{label}_ols_baseline"),
        ("OLS",   True,  f"{label}_ols_sentiment"),
        ("Ridge", False, f"{label}_ridge_baseline"),
        ("Ridge", True,  f"{label}_ridge_sentiment"),
    ]:
        r = tm.run_variant(df, model_type, include_sent, train_mask, test_mask)
        tr, te = r["train"], r["test"]
        row = {
            "variant": key,
            "model": model_type,
            "sentiment": include_sent,
            "alpha": r.get("alpha"),
            "train_RMSE": tr["RMSE"],  "test_RMSE": te["RMSE"],
            "train_MAE":  tr["MAE"],   "test_MAE":  te["MAE"],
            "train_dir_acc": tr["dir_acc"], "test_dir_acc": te["dir_acc"],
            "test_n": te["n"],
        }
        if model_type == "OLS" and include_sent:
            p, pv = r["params"], r["pvalues"]
            row["mean_sent_coef"] = float(p.get("mean_sentiment", np.nan))
            row["mean_sent_pvalue"] = float(pv.get("mean_sentiment", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    handoff = ROOT / "data" / "handoff"
    processed = p["data"]["processed"]
    tables = p["outputs"]["tables"]
    tables.mkdir(parents=True, exist_ok=True)

    scored_path = processed / "posts_scored_finbert.parquet"
    if scored_path.exists():
        print(f"reusing cached {scored_path.name}")
        scored = pd.read_parquet(scored_path)
    else:
        posts = load_posts(handoff)
        print(f"scoring {len(posts)} posts with {MODEL_ID} ...")
        probs = score_with_finbert(posts["text_for_sentiment"].tolist())
        scored = posts[["id", "ticker", "event_id", "rdq", "post_dt", "has_real_body"]].copy()
        scored["fb_p_pos"] = probs[:, 0]
        scored["fb_p_neu"] = probs[:, 1]
        scored["fb_p_neg"] = probs[:, 2]
        scored["fb_score"] = probs[:, 0] - probs[:, 2]
        scored.to_parquet(scored_path, index=False, compression="zstd")
        print(f"wrote {scored_path}")

    print(f"\nscored posts: {len(scored)}  (unique events: {scored['event_id'].nunique()})")
    print(f"fb_score mean = {scored['fb_score'].mean():+.4f}   std = {scored['fb_score'].std():.4f}")
    print(f"pos frac = {(scored['fb_score'] > POS_THRESH).mean():.3f}   "
          f"neg frac = {(scored['fb_score'] < NEG_THRESH).mean():.3f}   "
          f"neu frac = {scored['fb_score'].between(NEG_THRESH, POS_THRESH).mean():.3f}")

    skeleton = pd.read_parquet(handoff / "event_modeling_skeleton.parquet")
    agg = aggregate_event_sentiment(scored)
    df = build_dataset(skeleton, agg)
    df.to_parquet(processed / "event_modeling_dataset_finbert.parquet", index=False)

    vader = pd.read_parquet(processed / "event_modeling_dataset.parquet")[
        ["event_id", "mean_sentiment", "median_sentiment", "std_sentiment",
         "frac_positive", "frac_negative"]
    ].rename(columns=lambda c: c if c == "event_id" else f"vader_{c}")
    cmp = df.merge(vader, on="event_id", how="left")
    corr_mean = cmp[["mean_sentiment", "vader_mean_sentiment"]].corr().iloc[0, 1]
    corr_fracpos = cmp[["frac_positive", "vader_frac_positive"]].corr().iloc[0, 1]
    corr_fracneg = cmp[["frac_negative", "vader_frac_negative"]].corr().iloc[0, 1]
    print(f"\nEvent-level correlation (FinBERT vs VADER):")
    print(f"  mean_sentiment     r = {corr_mean:+.3f}")
    print(f"  frac_positive      r = {corr_fracpos:+.3f}")
    print(f"  frac_negative      r = {corr_fracneg:+.3f}")

    print("\n== OLS + Ridge with FinBERT sentiment ==")
    metrics = run_models(df, "finbert")
    metrics.to_csv(tables / "probe_finbert_metrics.csv", index=False)
    print(metrics[[
        "variant", "alpha", "test_RMSE", "test_MAE", "test_dir_acc",
        "mean_sent_coef", "mean_sent_pvalue",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print(f"\nwrote: {tables / 'probe_finbert_metrics.csv'}")


if __name__ == "__main__":
    main()
