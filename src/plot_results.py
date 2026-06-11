"""run_experiment.py が出力した CSV を読み込み、結果をグラフ化する。

生成する図 (result/ 以下):
  1. pvalue_hist.pdf     : 手法 × データ種別ごとの p 値ヒストグラム
                          (null なら一様分布に近いほど妥当)
  2. pvalue_qq.pdf       : p 値の QQ プロット (一様分布との比較)
  3. fpr.pdf            : 有意水準 0.05 での誤検出率 (FPR) の棒グラフ
  4. time.pdf          : 平均計算時間の棒グラフ (対数軸)

使い方:
  python src/plot_results.py --csv result/experiment_results.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]
DATA_TYPES = ["iid", "correlated"]


def plot_hist(df, outdir):
    """手法 × データ種別の p 値ヒストグラム (格子状)。"""
    fig, axes = plt.subplots(
        len(METHODS), len(DATA_TYPES),
        figsize=(8, 9), sharex=True, sharey=True,
    )
    for r, method in enumerate(METHODS):
        for c, dt in enumerate(DATA_TYPES):
            ax = axes[r, c]
            sub = df[(df.method == method) & (df.data_type == dt)]
            ax.hist(sub.p_value, bins=20, range=(0, 1),
                    color="steelblue", edgecolor="white")
            # 一様分布の期待度数を点線で表示
            ax.axhline(len(sub) / 20, color="red", ls="--", lw=1)
            if r == 0:
                ax.set_title(dt)
            if c == 0:
                ax.set_ylabel(f"{method}\nfrequency")
            if r == len(METHODS) - 1:
                ax.set_xlabel("p-value")
    fig.suptitle("p-value distribution")
    fig.tight_layout()
    path = os.path.join(outdir, "pvalue_hist.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def plot_qq(df, outdir):
    """p 値の経験分布 vs 一様分布 の QQ プロット。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for method in METHODS:
        for dt in DATA_TYPES:
            sub = df[(df.method == method) & (df.data_type == dt)]
            if sub.empty:
                continue
            p_sorted = np.sort(sub.p_value.values)
            n = len(p_sorted)
            expected = (np.arange(1, n + 1) - 0.5) / n
            ax.plot(expected, p_sorted, marker=".", ls="-",
                    label=f"{method} / {dt}")
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
    ax.set_xlabel("uniform quantile")
    ax.set_ylabel("observed p-value quantile")
    ax.set_title("QQ plot of p-values vs Uniform(0,1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "pvalue_qq.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def _grouped_bar(df, value_fn, ylabel, title, path, logy=False, hline=None):
    """手法を x 軸、データ種別を色分けした棒グラフの共通処理。"""
    x = np.arange(len(METHODS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, dt in enumerate(DATA_TYPES):
        vals = []
        for method in METHODS:
            sub = df[(df.method == method) & (df.data_type == dt)]
            vals.append(value_fn(sub) if len(sub) else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=dt)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="red", ls="--", lw=1, label=f"target {hline}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def plot_fpr(df, outdir, alpha=0.05):
    """有意水準 alpha での誤検出率 (p < alpha の割合)。"""
    _grouped_bar(
        df,
        value_fn=lambda s: float((s.p_value < alpha).mean()),
        ylabel=f"FPR (p < {alpha})",
        title=f"False positive rate at alpha={alpha}",
        path=os.path.join(outdir, "fpr.pdf"),
        hline=alpha,
    )


def plot_time(df, outdir):
    """平均計算時間 (対数軸)。"""
    _grouped_bar(
        df,
        value_fn=lambda s: float(s.time.mean()),
        ylabel="mean computation time [s]",
        title="p-value computation time",
        path=os.path.join(outdir, "time.pdf"),
        logy=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/experiment_results.csv")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    plot_hist(df, args.outdir)
    plot_qq(df, args.outdir)
    plot_fpr(df, args.outdir, args.alpha)
    plot_time(df, args.outdir)

    # 数値サマリも表示
    summary = df.groupby(["method", "data_type"]).agg(
        n=("p_value", "size"),
        mean_p=("p_value", "mean"),
        fpr=("p_value", lambda s: float((s < args.alpha).mean())),
        mean_time=("time", "mean"),
    )
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
