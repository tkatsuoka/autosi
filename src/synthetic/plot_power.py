"""検出力 (power) 実験の結果を (手法 × データ種別) ごとに折れ線グラフ化する。

横軸: 信号の傾き Δ、縦軸: power (= p 値 < α の割合)。
(手法, データ種別) の組ごとに別ファイル (result/power_<method>_<data_type>.pdf)
を出力し、proposed / w/o-pp / bonferroni / data splitting の
各 p 値の power 曲線を重ねて描く (色は FPR プロットと対応。
naive は FPR を制御しない invalid な手法のため power には描かない)。

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


def _power_curve(sub, p_col, flag_col, alpha, n_pvalues, label, method, data_type):
    """1 本分の power 曲線 (Δ -> 棄却率) を集計して返す。

    実際に信号が載った特徴量が選ばれた試行のみで power を測る。
    null 特徴量 (真の係数 0) が選ばれて棄却された試行は Type I error なので、
    power の分母・分子どちらからも除外する (= 条件付き検出力)。
    """
    if flag_col in sub.columns:
        n_total = len(sub)
        sub = sub[sub[flag_col]]
        # 動的モードで目標本数の充足後に SI 推論を省略した行は p 値が
        # 空なので除く (DS 収集専用フェーズの行。除外はシード順の打ち切り
        # と同じで、p 値の値には依存しないためバイアスは生じない)
        sub = sub[sub[p_col].notna()]
        print(f"{method} ({data_type}) [{label}]: 有効 p 値 "
              f"{len(sub)}/{n_total} 件で power を算出")
    else:
        print(f"警告: {method} ({data_type}) に {flag_col} 列が無い (旧形式CSV)。"
              "全試行で power を算出します。run_experiment.py で再計算してください。")

    # 信号選択で絞った後、条件 (Δ) ごとに seed 順で先頭 n_pvalues 件だけ使う
    sub = limit_per_group(sub, ["signal"], n_pvalues)

    # Δ ごとに power = P(p 値 < α | 信号特徴量を選択) を集計
    return (
        sub.assign(reject=sub[p_col] < alpha)
        .groupby("signal")["reject"].mean()
        .sort_index()
    )


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None):
    """(手法, データ種別) 1 組の power 折れ線グラフ (横軸 Δ) を描く。"""
    # (凡例ラベル, p 値列, 信号選択フラグ列, 色番号)。色は plot_fpr と揃える
    # (naive = C2 は FPR を制御しない invalid な手法のため power には描かない)。
    # データ分割法は前半のみで選択するため、条件付けに使うフラグ列も別
    # (ds_is_signal)。
    curves = [
        ("proposed", "p_value", "is_signal", 0),
        ("w/o-pp", "oc_p_value", "is_signal", 1),
        ("bonferroni", "bonferroni_p_value", "is_signal", 3),
        ("data splitting", "ds_p_value", "ds_is_signal", 4),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, p_col, flag_col, i in curves:
        # 旧形式 CSV に列が無い場合は飛ばす (色番号は固定のまま)
        if p_col not in sub.columns:
            continue
        power = _power_curve(sub, p_col, flag_col, alpha, n_pvalues,
                             label, method, data_type)
        ax.plot(power.index, power.values, marker="o", color=f"C{i}",
                label=label)
    ax.axhline(alpha, color="gray", ls="--", lw=1, label=f"alpha={alpha}")
    ax.set_xlabel("signal Δ")
    ax.set_ylabel("power")
    # ax.set_title(f"{method} ({data_type})")
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
