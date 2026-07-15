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

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.merge_results import load_results, limit_per_group


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None):
    """(手法, データ種別) 1 組の FPR 折れ線グラフ (横軸 n, 線 1 本) を描く。"""
    # 条件 (サンプル数) ごとに seed 順で先頭 n_pvalues 件だけ使う
    sub = limit_per_group(sub, ["n_samples"], n_pvalues)
    # サンプル数ごとに type I error rate = P(p 値 < α) を集計
    p_cols = {
        "p_value": "proposed",
        "oc_p_value": "o/w-pp",
        "naive_p_value": "naive",
        "bonferroni_p_value": "bonferroni",
        "ds_p_value": "data splitting",
    }

    fig, ax = plt.subplots(figsize=(7, 5)) # 凡例スペースのために少し横幅を広げています

    # カラーサイクルのインデックス用
    for i, (col, label) in enumerate(p_cols.items()):
        # データに該当する列が存在しない場合の安全対策
        if col not in sub.columns:
            continue
            
        # 各 p 値ごとに type I error rate = P(p 値 < α) を集計
        rate = (
            sub.assign(reject=sub[col] < alpha)
            .groupby("n_samples")["reject"].mean()
            .sort_index()
        )
        
        # 折れ線を描画 (C0, C1, C2... と自動で色が変わります)
        ax.plot(rate.index, rate.values, marker="o", color=f"C{i}", label=label)

    ax.axhline(alpha, color="gray", ls="--", lw=1, label=f"alpha={alpha}")
    ax.set_xlabel("number of samples n")
    ax.set_ylabel("type I error rate")
    ax.set_title(f"{method} ({data_type})")
    ax.set_ylim(0, 1.0)
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


if __name__ == "__main__":
    main()
