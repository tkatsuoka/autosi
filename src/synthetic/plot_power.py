"""検出力 (power) 実験の結果を (手法 × データ種別) ごとに折れ線グラフ化する。

横軸: 信号の傾き Δ、縦軸: power (= p 値 < α の割合)。
(手法, データ種別) の組ごとに完全に別ファイル
(result/power_<method>_<data_type>.pdf) を 1 本の折れ線で出力する。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/synthetic/plot_power.py --csv result/power/lasso_iid_d0.1
  python src/synthetic/plot_power.py --csv result/power
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.merge_results import load_results, limit_per_group


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None):
    """(手法, データ種別) 1 組の power 折れ線グラフ (横軸 Δ, 線 1 本) を描く。"""
    # 実際に信号が載った特徴量が選ばれた試行のみで power を測る。
    # null 特徴量 (真の係数 0) が選ばれて棄却された試行は Type I error なので、
    # power の分母・分子どちらからも除外する (= 条件付き検出力)。
    if "is_signal" in sub.columns:
        n_total = len(sub)
        sub = sub[sub["is_signal"]]
        print(f"{method} ({data_type}): 信号特徴量が選ばれた試行 "
              f"{len(sub)}/{n_total} のみで power を算出")
    else:
        print(f"警告: {method} ({data_type}) に is_signal 列が無い (旧形式CSV)。"
              "全試行で power を算出します。run_experiment.py で再計算してください。")

    # is_signal で絞った後、条件 (Δ) ごとに seed 順で先頭 n_pvalues 件だけ使う
    sub = limit_per_group(sub, ["signal"], n_pvalues)

    # Δ ごとに power = P(p 値 < α | 信号特徴量を選択) を集計
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
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="各条件で使用する p 値の個数 (is_signal で絞った後、"
                             "seed 順の先頭から。既定: 全件)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # (手法, データ種別) の組ごとに 1 枚ずつ出力
    for (method, data_type), sub in df.groupby(["method", "data_type"]):
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues)


if __name__ == "__main__":
    main()
