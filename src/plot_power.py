"""検出力 (power) 実験の結果を (手法 × データ種別) ごとに折れ線グラフ化する。

横軸: 信号の傾き Δ、縦軸: power (= p 値 < α の割合)。
(手法, データ種別) の組ごとに完全に別ファイル
(result/power_<method>_<data_type>.pdf) を 1 本の折れ線で出力する。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/plot_power.py --csv result/power/lasso_iid_d0.1
  python src/plot_power.py --csv result/power
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.merge_results import load_results


def plot_one(sub, method, data_type, alpha, outdir):
    """(手法, データ種別) 1 組の power 折れ線グラフ (横軸 Δ, 線 1 本) を描く。"""
    # Δ ごとに power = P(p 値 < α) を集計
    power = (
        sub.assign(reject=sub.p_value < alpha)
        .groupby("signal")["reject"].mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(power.index, power.values, marker="o", color="C0")
    ax.axhline(alpha, color="gray", ls="--", lw=1, label=f"alpha={alpha}")
    ax.set_xlabel("signal Δ")
    ax.set_ylabel("power")
    ax.set_title(f"{method} ({data_type})")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, f"power_{method}_{data_type}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/power",
                        help="CSV ファイル、またはCSVを含むフォルダ")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # (手法, データ種別) の組ごとに 1 枚ずつ出力
    for (method, data_type), sub in df.groupby(["method", "data_type"]):
        plot_one(sub, method, data_type, args.alpha, args.outdir)


if __name__ == "__main__":
    main()
