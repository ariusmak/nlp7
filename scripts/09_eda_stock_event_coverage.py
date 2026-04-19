from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import paths, universe


AMBIGUOUS = {"AMD", "BA", "DIS", "GS", "MU"}

BL = r"(?<![A-Za-z0-9_])"
BR = r"(?![A-Za-z0-9_])"


def _ticker_aliases(universe_cfg: dict) -> dict[str, list[str]]:
    return {row["ticker"]: row["aliases"] for row in universe_cfg["tickers"]}


def _classify_match(text: str, ticker: str, aliases: list[str]) -> str:
    t = str(text or "")
    if re.search(BL + r"\$" + ticker + BR, t, re.IGNORECASE):
        return "cashtag"
    alias_terms = [a for a in aliases if a.upper() != ticker]
    if alias_terms:
        alias_re = BL + "(?:" + "|".join(re.escape(a) for a in alias_terms) + ")" + BR
        if re.search(alias_re, t, re.IGNORECASE):
            return "alias"
    if re.search(BL + ticker + BR, t, re.IGNORECASE):
        return "bare_ticker"
    return "unknown"


def fig_coverage_per_ticker(events: pd.DataFrame, out_path: Path) -> None:
    counts = events.copy()
    counts["bucket"] = pd.cut(
        counts["post_count"],
        bins=[-0.5, 0.5, 9.5, np.inf],
        labels=["zero", "thin (1-9)", "healthy (>=10)"],
    )
    pivot = counts.pivot_table(index="ticker", columns="bucket", values="event_id", aggfunc="count", observed=False).fillna(0)
    pivot = pivot.reindex(
        counts.groupby("ticker")["post_count"].mean().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pivot))
    colors = {"healthy (>=10)": "#2a9d8f", "thin (1-9)": "#e9c46a", "zero": "#e76f51"}
    for b in ["healthy (>=10)", "thin (1-9)", "zero"]:
        if b in pivot.columns:
            vals = pivot[b].values
            ax.bar(pivot.index, vals, bottom=bottom, label=b, color=colors[b])
            bottom = bottom + vals
    ax.set_title("Earnings events per ticker by Reddit coverage (10-day pre-earnings window)")
    ax.set_ylabel("events")
    ax.set_xlabel("ticker")
    ax.legend(title="post_count bucket", loc="upper right")
    for i, t in enumerate(pivot.index):
        ax.text(i, bottom[i] + 0.2, f"n={int(pivot.loc[t].sum())}", ha="center", va="bottom", fontsize=8, color="#333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_post_count_box(events: pd.DataFrame, out_path: Path) -> None:
    order = events.groupby("ticker")["post_count"].mean().sort_values(ascending=False).index.tolist()
    data = [events.loc[events["ticker"] == t, "post_count"].values for t in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, showfliers=True, patch_artist=True,
               boxprops=dict(facecolor="#ddeef2", linewidth=1), medianprops=dict(color="#264653"))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=0)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Distribution of matched posts per event, by ticker (symlog y)")
    ax.set_ylabel("post_count (symlog)")
    ax.set_xlabel("ticker")
    ax.axhline(10, color="#e76f51", linestyle=":", linewidth=1, label="thin cutoff (<10)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_sentiment_hist_raw_vs_adapted(posts: pd.DataFrame, out_path: Path) -> None:
    raw = posts.loc[posts["text_clean"].str.len() > 0, "raw_compound"]
    adp = posts.loc[posts["text_clean"].str.len() > 0, "adp_compound"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1, 1, 41)
    ax.hist(raw, bins=bins, alpha=0.5, label=f"raw VADER (N={len(raw):,})", color="#6c757d", edgecolor="none")
    ax.hist(adp, bins=bins, alpha=0.5, label=f"adapted (+WSB lex) (N={len(adp):,})", color="#2a9d8f", edgecolor="none")
    ax.axvline(0.05, color="#888", linestyle=":", linewidth=1)
    ax.axvline(-0.05, color="#888", linestyle=":", linewidth=1)
    ax.set_title("Post-level VADER compound distribution: raw vs adapted")
    ax.set_xlabel("compound score")
    ax.set_ylabel("posts")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_event_sentiment_vs_target(modeling_df: pd.DataFrame, out_path: Path) -> None:
    df = modeling_df[modeling_df["post_count"] >= 10].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    tickers = sorted(df["ticker"].unique())
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(tickers):
        sub = df[df["ticker"] == t]
        ax.scatter(sub["mean_sentiment"], sub["target_ret3"], s=22, alpha=0.7,
                   color=cmap(i % 20), label=t)
    if len(df) >= 2:
        x = df["mean_sentiment"].values
        y = df["target_ret3"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#264653", linewidth=1.5,
                    linestyle="--", label=f"OLS fit (slope={slope:.3f})")
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
    else:
        corr = float("nan")
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.set_title(f"Event mean_sentiment vs 3-day post-earnings return  (N={len(df)} events >=10 posts; Pearson r = {corr:.3f})")
    ax.set_xlabel("event mean_sentiment (adapted VADER)")
    ax.set_ylabel("target_ret3")
    ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_target_distribution(modeling_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(modeling_df["target_ret3"], bins=50, color="#264653", alpha=0.85)
    ax.axvline(0, color="#e76f51", linewidth=1)
    mu = modeling_df["target_ret3"].mean()
    sd = modeling_df["target_ret3"].std()
    ax.axvline(mu, color="#f4a261", linewidth=1, linestyle="--", label=f"mean={mu:.3f}")
    ax.set_title(f"3-day post-earnings return distribution (N={len(modeling_df)}, std={sd:.3f})")
    ax.set_xlabel("target_ret3")
    ax.set_ylabel("events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sample_posts_per_ticker(posts_long: pd.DataFrame, n: int = 5, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t, g in posts_long.groupby("ticker"):
        take = min(n, len(g))
        if take == 0:
            continue
        idx = rng.choice(len(g), size=take, replace=False)
        s = g.iloc[idx][["ticker", "id", "post_date", "num_matched", "multi_ticker_flag", "title"]]
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out["title"] = out["title"].astype(str).str.slice(0, 160)
    return out


def ambiguous_audit(posts_long: pd.DataFrame, aliases_map: dict[str, list[str]],
                    n_per_ticker: int = 15, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in sorted(AMBIGUOUS):
        g = posts_long[posts_long["ticker"] == t].copy()
        if g.empty:
            continue
        combined = (g["title"].fillna("").astype(str) + " " + g["selftext"].fillna("").astype(str))
        g = g.assign(
            match_type=[_classify_match(text, t, aliases_map[t]) for text in combined]
        )
        bare = g[g["match_type"] == "bare_ticker"]
        if bare.empty:
            continue
        take = min(n_per_ticker, len(bare))
        idx = rng.choice(len(bare), size=take, replace=False)
        s = bare.iloc[idx][["ticker", "id", "post_date", "num_matched", "multi_ticker_flag",
                            "match_type", "title", "selftext"]]
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["title"] = out["title"].astype(str).str.slice(0, 200)
    out["selftext"] = out["selftext"].astype(str).str.slice(0, 300)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = paths()
    u = universe()
    figs_dir = p["outputs"]["figures"]
    tbl_dir = p["outputs"]["tables"]
    figs_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    modeling = pd.read_parquet(p["data"]["processed"] / "event_modeling_dataset.parquet")
    events_coverage = pd.read_parquet(p["data"]["interim"] / "event_post_counts.parquet")
    posts_scored = pd.read_parquet(p["data"]["processed"] / "posts_scored.parquet")
    posts_long = pd.read_parquet(p["data"]["interim"] / "posts_matched_long.parquet")

    print(f"modeling rows      : {len(modeling):,}")
    print(f"event coverage rows: {len(events_coverage):,}")
    print(f"scored posts       : {len(posts_scored):,}")
    print(f"matched posts long : {len(posts_long):,}")
    print()

    fig1 = figs_dir / "01_coverage_per_ticker.png"
    fig_coverage_per_ticker(events_coverage, fig1)
    print(f"wrote: {fig1}")

    fig2 = figs_dir / "02_post_count_box_per_ticker.png"
    fig_post_count_box(events_coverage, fig2)
    print(f"wrote: {fig2}")

    fig3 = figs_dir / "03_sentiment_hist_raw_vs_adapted.png"
    fig_sentiment_hist_raw_vs_adapted(posts_scored, fig3)
    print(f"wrote: {fig3}")

    fig4 = figs_dir / "04_event_sentiment_vs_target.png"
    fig_event_sentiment_vs_target(modeling, fig4)
    print(f"wrote: {fig4}")

    fig5 = figs_dir / "05_target_distribution.png"
    fig_target_distribution(modeling, fig5)
    print(f"wrote: {fig5}")

    sample_tbl = sample_posts_per_ticker(posts_long, n=5)
    sample_path = tbl_dir / "eda_sample_posts_per_ticker.csv"
    sample_tbl.to_csv(sample_path, index=False)
    print(f"wrote: {sample_path}  ({len(sample_tbl)} rows)")

    aliases_map = _ticker_aliases(u)
    audit_tbl = ambiguous_audit(posts_long, aliases_map, n_per_ticker=15)
    audit_path = tbl_dir / "eda_ambiguous_audit_bare_ticker.csv"
    audit_tbl.to_csv(audit_path, index=False)
    print(f"wrote: {audit_path}  ({len(audit_tbl)} rows)")

    print()
    corr_df = modeling[modeling["post_count"] >= 10].copy()
    feats_for_corr = ["post_count", "mean_sentiment", "median_sentiment", "std_sentiment",
                      "frac_positive", "frac_negative",
                      "ret5_pre", "ret20_pre", "vol20_pre", "abvol_pre"]
    corr = corr_df[feats_for_corr + ["target_ret3"]].corr()["target_ret3"].drop("target_ret3")
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"pearson corr with target_ret3 (N={len(corr_df)} events with >=10 posts):")
    print(corr.round(4).to_string())
    corr.to_csv(tbl_dir / "eda_feature_target_correlations.csv")
    print(f"wrote: {tbl_dir / 'eda_feature_target_correlations.csv'}")


if __name__ == "__main__":
    main()
