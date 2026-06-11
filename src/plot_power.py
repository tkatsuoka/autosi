"""検出力 (power) 実験の結果を作図する。

横軸を信号の傾き Δ、縦軸を power (= p 値 < α の割合) として、
手法ごと・データ種別ごとに曲線を描く。Δ が大きいほど power が
1 に近づくのが期待される挙動。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/plot_power.py --csv result/power_results.csv
  python src/plot_power.py --csv result/power
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.merge_results import load_results

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]
DATA_TYPES = ["iid", "correlated"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/power",
                        help="CSV ファイル、またはCSVを含むフォルダ")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # 各 (手法, データ種別, Δ) で power = P(p 値 < α) を集計
    df["reject"] = df.p_value < args.alpha
    grouped = (
        df.groupby(["method", "data_type", "signal"])["reject"]
        .mean()
        .reset_index()
        .rename(columns={"reject": "power"})
    )

    # データ種別ごとにサブプロット。手法を曲線で重ねる。
    fig, axes = plt.subplots(1, len(DATA_TYPES), figsize=(11, 5), sharey=True)
    for ax, dt in zip(np.atleast_1d(axes), DATA_TYPES):
        for method in METHODS:
            sub = grouped[(grouped.method == method) & (grouped.data_type == dt)]
            sub = sub.sort_values("signal")
            if sub.empty:
                continue
            ax.plot(sub.signal, sub.power, marker="o", label=method)
        ax.axhline(args.alpha, color="gray", ls="--", lw=1,
                   label=f"alpha={args.alpha}")
        ax.set_title(dt)
        ax.set_xlabel("signal Δ")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("power (P[p-value < alpha])")
    axes[0].legend(fontsize=9)
    fig.suptitle("Power vs signal strength")
    fig.tight_layout()
    path = os.path.join(args.outdir, "power.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
    print("\n" + grouped.to_string(index=False))


if __name__ == "__main__":
    main()
