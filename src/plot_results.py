"""Type I error (FPR) 実験の結果を (手法 × データ種別) ごとに折れ線グラフ化する。

横軸: サンプル数 n、縦軸: type I error rate (= p 値 < α の割合)。
(手法, データ種別) の組ごとに完全に別ファイル
(result/fpr_<method>_<data_type>.pdf) を 1 本の折れ線で出力する。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/plot_results.py --csv result/fpr/lasso_iid
  python src/plot_results.py --csv result/fpr
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.merge_results import load_results


def plot_one(sub, method, data_type, alpha, outdir):
    """(手法, データ種別) 1 組の FPR 折れ線グラフ (横軸 n, 線 1 本) を描く。"""
    # サンプル数ごとに type I error rate = P(p 値 < α) を集計
    rate = (
        sub.assign(reject=sub.p_value < alpha)
        .groupby("n_samples")["reject"].mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rate.index, rate.values, marker="o", color="C0")
    ax.axhline(alpha, color="gray", ls="--", lw=1, label=f"alpha={alpha}")
    ax.set_xlabel("number of samples n")
    ax.set_ylabel("type I error rate")
    ax.set_title(f"{method} ({data_type})")
    ax.set_ylim(0, max(2 * alpha, 0.11))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, f"fpr_{method}_{data_type}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/fpr",
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
