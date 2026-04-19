from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.paths import paths


BASELINE_FEATS = ["ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre"]
SENTIMENT_FEATS = ["post_count", "mean_sentiment", "median_sentiment",
                   "std_sentiment", "frac_positive", "frac_negative"]
ALPHA_GRID = np.logspace(-3, 5, 33)


def build_design(df: pd.DataFrame, include_sentiment: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric = list(BASELINE_FEATS)
    if include_sentiment:
        numeric = numeric + list(SENTIMENT_FEATS)
    tk = pd.get_dummies(df["ticker"], prefix="tk", drop_first=True).astype(float)
    yr = pd.get_dummies(df["calendar_year"], prefix="yr", drop_first=True).astype(float)
    dummy_cols = tk.columns.tolist() + yr.columns.tolist()
    X = pd.concat([df[numeric].astype(float), tk, yr], axis=1)
    return X, numeric, dummy_cols


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    dir_acc = float(np.mean(sign_pred == sign_true))
    return {"RMSE": rmse, "MAE": mae, "dir_acc": dir_acc, "n": int(len(y_true))}


def fit_ols(X_train: pd.DataFrame, y_train: pd.Series, numeric_cols: list[str]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, SimpleImputer]:
    imp = SimpleImputer(strategy="mean")
    X_imp = X_train.copy()
    X_imp[numeric_cols] = imp.fit_transform(X_train[numeric_cols])
    X_imp = sm.add_constant(X_imp, has_constant="add")
    model = sm.OLS(y_train, X_imp).fit()
    return model, imp


def predict_ols(model, imp: SimpleImputer, X: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    X_imp = X.copy()
    X_imp[numeric_cols] = imp.transform(X[numeric_cols])
    X_imp = sm.add_constant(X_imp, has_constant="add")
    X_imp = X_imp.reindex(columns=model.model.exog_names, fill_value=0.0)
    return np.asarray(model.predict(X_imp))


def make_ridge_pipeline(numeric_cols: list[str], dummy_cols: list[str], alpha: float) -> Pipeline:
    ct = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                              ("sc", StandardScaler())]), numeric_cols),
            ("pass", "passthrough", dummy_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", ct), ("ridge", Ridge(alpha=alpha, solver="auto"))])


def walk_forward_alpha(df: pd.DataFrame, include_sentiment: bool) -> tuple[float, pd.DataFrame]:
    X_all, numeric, dummies = build_design(df, include_sentiment)
    y_all = df["target_ret3"].astype(float).values
    years = df["calendar_year"].values

    folds = [
        (list(range(2016, 2019)), 2019),
        (list(range(2016, 2020)), 2020),
        (list(range(2016, 2021)), 2021),
    ]
    cv_rows = []
    for train_years, val_year in folds:
        tr_mask = np.isin(years, train_years)
        va_mask = years == val_year
        X_tr = X_all[tr_mask]
        y_tr = y_all[tr_mask]
        X_va = X_all[va_mask]
        y_va = y_all[va_mask]
        for a in ALPHA_GRID:
            pipe = make_ridge_pipeline(numeric, dummies, alpha=a)
            pipe.fit(X_tr, y_tr)
            yhat = pipe.predict(X_va)
            m = _metrics(y_va, yhat)
            cv_rows.append({"alpha": float(a), "val_year": int(val_year), **m})

    cv_df = pd.DataFrame(cv_rows)
    agg = (cv_df.groupby("alpha")
                .apply(lambda g: np.sqrt(np.sum((g["RMSE"] ** 2) * g["n"]) / g["n"].sum()),
                       include_groups=False)
                .rename("cv_rmse").reset_index())
    best_alpha = float(agg.sort_values("cv_rmse").iloc[0]["alpha"])
    return best_alpha, cv_df.merge(agg, on="alpha", how="left")


def run_variant(df: pd.DataFrame, model_type: str, include_sentiment: bool,
                train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    X_all, numeric, dummies = build_design(df, include_sentiment)
    y_all = df["target_ret3"].astype(float).values

    X_tr = X_all[train_mask]
    y_tr = y_all[train_mask]
    X_te = X_all[test_mask]
    y_te = y_all[test_mask]

    result = {"model": model_type, "sentiment": include_sentiment}

    if model_type == "OLS":
        ols, imp = fit_ols(X_tr, pd.Series(y_tr), numeric)
        yhat_tr = predict_ols(ols, imp, X_tr, numeric)
        yhat_te = predict_ols(ols, imp, X_te, numeric)
        result["train"] = _metrics(y_tr, yhat_tr)
        result["test"] = _metrics(y_te, yhat_te)
        result["r2_train"] = float(ols.rsquared)
        result["r2_train_adj"] = float(ols.rsquared_adj)
        result["predictions_test"] = yhat_te
        result["params"] = ols.params
        result["pvalues"] = ols.pvalues
        result["alpha"] = None
        result["ols_result"] = ols
        return result

    if model_type == "Ridge":
        best_alpha, cv_full = walk_forward_alpha(df.loc[train_mask].reset_index(drop=True), include_sentiment)
        pipe = make_ridge_pipeline(numeric, dummies, alpha=best_alpha)
        pipe.fit(X_tr, y_tr)
        yhat_tr = pipe.predict(X_tr)
        yhat_te = pipe.predict(X_te)
        result["train"] = _metrics(y_tr, yhat_tr)
        result["test"] = _metrics(y_te, yhat_te)
        result["predictions_test"] = yhat_te
        result["alpha"] = best_alpha
        result["cv_detail"] = cv_full
        ridge = pipe.named_steps["ridge"]
        prep = pipe.named_steps["prep"]
        feat_names = numeric + dummies
        result["params"] = pd.Series(ridge.coef_, index=feat_names)
        result["intercept"] = float(ridge.intercept_)
        scales = prep.named_transformers_["num"].named_steps["sc"].scale_
        result["num_scale"] = pd.Series(scales, index=numeric)
        return result

    raise ValueError(model_type)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    processed = p["data"]["processed"]
    tables_dir = p["outputs"]["tables"]
    figs_dir = p["outputs"]["figures"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed / "event_modeling_dataset.parquet").reset_index(drop=True)
    df = df.sort_values(["rdq", "ticker"]).reset_index(drop=True)

    train_mask = df["calendar_year"].isin(range(2016, 2022)).values
    test_mask = df["calendar_year"].isin([2022, 2023]).values
    print(f"train events: {int(train_mask.sum())}  (2016-2021)")
    print(f"test  events: {int(test_mask.sum())}   (2022-2023-03-28)")
    print()

    variants = [
        ("OLS", False, "ols_baseline"),
        ("OLS", True, "ols_sentiment"),
        ("Ridge", False, "ridge_baseline"),
        ("Ridge", True, "ridge_sentiment"),
    ]
    results: dict[str, dict] = {}
    metric_rows = []
    for model_type, include_sent, key in variants:
        print(f"== fitting {key} ==")
        r = run_variant(df, model_type, include_sent, train_mask, test_mask)
        results[key] = r
        tr = r["train"]
        te = r["test"]
        metric_rows.append({
            "variant": key,
            "model": model_type,
            "sentiment": include_sent,
            "alpha": r.get("alpha"),
            "train_RMSE": tr["RMSE"],
            "train_MAE": tr["MAE"],
            "train_dir_acc": tr["dir_acc"],
            "train_n": tr["n"],
            "test_RMSE": te["RMSE"],
            "test_MAE": te["MAE"],
            "test_dir_acc": te["dir_acc"],
            "test_n": te["n"],
        })
        print(f"  train: RMSE={tr['RMSE']:.4f} MAE={tr['MAE']:.4f} dir_acc={tr['dir_acc']:.3f}")
        print(f"  test : RMSE={te['RMSE']:.4f} MAE={te['MAE']:.4f} dir_acc={te['dir_acc']:.3f}")
        if r.get("alpha") is not None:
            print(f"  alpha (CV)={r['alpha']:.4f}")
        print()

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables_dir / "model_metrics.csv", index=False)

    for key, r in results.items():
        if "params" in r:
            params = r["params"].copy()
            col = {"coef": params}
            if "pvalues" in r:
                col["pvalue"] = r["pvalues"]
            if "num_scale" in r:
                col["num_scale"] = r["num_scale"]
            coefs = pd.DataFrame(col)
            coefs.index.name = "feature"
            coefs.to_csv(tables_dir / f"coefficients_{key}.csv")

    ridge_cv_rows = []
    for key in ("ridge_baseline", "ridge_sentiment"):
        r = results[key]
        cv = r["cv_detail"].copy()
        cv["variant"] = key
        ridge_cv_rows.append(cv)
    pd.concat(ridge_cv_rows, ignore_index=True).to_csv(tables_dir / "ridge_cv_detail.csv", index=False)

    preds_df = df.loc[test_mask, ["ticker", "event_id", "rdq", "calendar_year", "target_ret3"]].copy()
    for key, r in results.items():
        preds_df[f"pred_{key}"] = r["predictions_test"]
    preds_df.to_parquet(processed / "model_predictions_test.parquet", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (_, _, key) in zip(axes.flatten(), variants):
        ax.scatter(preds_df["target_ret3"], preds_df[f"pred_{key}"], s=20, alpha=0.75,
                   color="#264653")
        lim = max(abs(preds_df["target_ret3"]).max(), abs(preds_df[f"pred_{key}"]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#e76f51", linewidth=1, linestyle="--")
        ax.axhline(0, color="#aaa", linewidth=0.6)
        ax.axvline(0, color="#aaa", linewidth=0.6)
        te = results[key]["test"]
        ax.set_title(f"{key}  (RMSE={te['RMSE']:.4f}, dir_acc={te['dir_acc']:.2f})")
        ax.set_xlabel("actual target_ret3")
        ax.set_ylabel("predicted")
    fig.suptitle("Test-set predictions vs actuals (2022–2023)")
    fig.tight_layout()
    fig.savefig(figs_dir / "06_predictions_vs_actual.png", dpi=150)
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
               boxprops=dict(facecolor="#ddeef2"), medianprops=dict(color="#264653"))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=0)
    ax.axhline(0, color="#e76f51", linewidth=1)
    ax.set_title("Test-set residual distribution by variant")
    ax.set_ylabel("residual (actual - predicted)")
    fig.tight_layout()
    fig.savefig(figs_dir / "07_residuals_by_variant.png", dpi=150)
    plt.close(fig)

    print("SUMMARY (test metrics):")
    print(metrics_df[["variant", "alpha", "test_RMSE", "test_MAE", "test_dir_acc"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print()
    print(f"wrote: {tables_dir / 'model_metrics.csv'}")
    print(f"wrote: {tables_dir / 'ridge_cv_detail.csv'}")
    print(f"wrote: {processed / 'model_predictions_test.parquet'}")
    print(f"wrote: {figs_dir / '06_predictions_vs_actual.png'}")
    print(f"wrote: {figs_dir / '07_residuals_by_variant.png'}")


if __name__ == "__main__":
    main()
