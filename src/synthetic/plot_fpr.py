"""Type I error (FPR) 実験の結果を (手法 × データ種別) ごとに折れ線グラフ化する。

横軸: サンプル数 n、縦軸: type I error rate (= p 値 < α の割合)。
(手法, データ種別) の組ごとに完全に別ファイル
(result/fpr_<method>_<data_type>.pdf) を 1 本の折れ線で出力する。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/synthetic/plot_fpr.py --csv result/fpr/lasso_iid
  python src/synthetic/plot_fpr.py --csv result/fpr
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.merge_results import load_results, limit_per_group
from src.common.plot_style import FIGSIZE, DPI, FONT_SIZE, line_style, save_legend


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None):
    """(手法, データ種別) 1 組の FPR 折れ線グラフ (横軸 n, 線 1 本) を描く。"""
    # 条件 (サンプル数) ごとに seed 順で先頭 n_pvalues 件だけ使う
    sub = limit_per_group(sub, ["n_samples"], n_pvalues)
    # (凡例ラベル, p 値列, zorder)。描画順 = 凡例順 = 色サイクル順で、
    # zorder は proposed が最前面になるように振る (plot_power と共通の並び)
    curves = [
        ("proposed", "p_value", 10),
        ("w/o-pp", "oc_p_value", 9),
        ("bonferroni", "bonferroni_p_value", 8),
        ("data splitting", "ds_p_value", 7),
        ("naive", "naive_p_value", 6),
    ]

    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    ns = sorted(sub["n_samples"].unique())
    ax.plot(ns, [alpha] * len(ns), linestyle="dashed", color="black")
    for label, col, z in curves:
        # データに該当する列が存在しない場合の安全対策 (旧形式 CSV)
        if col not in sub.columns:
            continue
        # 各 p 値ごとに type I error rate = P(p 値 < α) を集計
        rate = (
            sub.assign(reject=sub[col] < alpha)
            .groupby("n_samples")["reject"].mean()
            .sort_index()
        )
        ax.plot(rate.index, rate.values, label=label, zorder=z,
                **line_style(label))

    ax.set_xticks(ns)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Type I Error Rate")
    # 凡例は各図には描かず、共通の legend.pdf (save_legend) を使う
    path = os.path.join(outdir, f"fpr_{method}_{data_type}.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/fpr",
                        help="CSV ファイル、またはCSVを含むフォルダ")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="各条件で使用する p 値の個数 (seed 順の先頭から。"
                             "既定: 全件)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # (手法, データ種別) の組ごとに 1 枚ずつ出力
    for (method, data_type), sub in df.groupby(["method", "data_type"]):
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues)

    # FPR の全図共通の凡例だけの PDF (naive を含む 5 手法)
    save_legend(os.path.join(args.outdir, "legend_fpr.pdf"))


if __name__ == "__main__":
    main()
