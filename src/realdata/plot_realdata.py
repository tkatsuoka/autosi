"""実データ実験の結果を論文 Table 1 形式の表に集計する。

各 (手法, サンプル数 n, データセット) の棄却率 P(p 値 < α) を
「proposed / w/o-pp / bonferroni / ds」の形で並べた表を出力する
(実データでの power 指標。naive は invalid のため表には出さないが、
realdata_summary.csv には全 p 値列の棄却率が入る)。

使い方 (CSV 単体でもフォルダでも可。フォルダは再帰的に全 CSV を読む):
  python src/realdata/plot_realdata.py --csv result/realdata
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common.merge_results import load_results, limit_per_group


def summarize(df, alpha):
    """(手法, n, データセット) ごとの棄却率を縦持ちで集計する。"""
    aggs = dict(
        trials=("p_value", "size"),
        proposed=("p_value", lambda s: (s < alpha).mean()),
        oc=("oc_p_value", lambda s: (s < alpha).mean()),
        naive=("naive_p_value", lambda s: (s < alpha).mean()),
        bonferroni=("bonferroni_p_value", lambda s: (s < alpha).mean()),
    )
    # データ分割法 (旧形式 CSV には列が無いので存在チェック)
    if "ds_p_value" in df.columns:
        aggs["ds"] = ("ds_p_value", lambda s: (s < alpha).mean())
    return (
        df.groupby(["method", "n_samples", "dataset"])
        .agg(**aggs)
        .reset_index()
    )


def table_legend(summary):
    """論文形式の表のセル凡例を返す (ds 列は旧形式 CSV に無いことがある)。"""
    legend = "proposed/w/o-pp/bonferroni"
    return legend + ("/ds" if "ds" in summary.columns else "")


def to_paper_table(sub):
    """1 手法分を「行 = n, 列 = データセット, セル = 棄却率の併記」に整形する。"""
    has_ds = "ds" in sub.columns

    def fmt(r):
        cell = f"{r.proposed:.2f}/{r.oc:.2f}/{r.bonferroni:.2f}"
        return cell + (f"/{r.ds:.2f}" if has_ds else "")

    cell = sub.assign(cell=sub.apply(fmt, axis=1))
    return cell.pivot(index="n_samples", columns="dataset", values="cell")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/realdata",
                        help="CSV ファイル、またはCSVを含むフォルダ")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="各条件で使用する p 値の個数 (seed 順の先頭から。"
                             "既定: 全件)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # 条件 (手法, n, データセット) ごとに seed 順で先頭 n_pvalues 件だけ使う
    df = limit_per_group(df, ["method", "n_samples", "dataset"], args.n_pvalues)

    summary = summarize(df, args.alpha)
    long_path = os.path.join(args.outdir, "realdata_summary.csv")
    summary.to_csv(long_path, index=False)
    print(f"saved {long_path}")

    # 手法ごとに論文 Table 1 形式の表を出力
    legend = table_legend(summary)
    for method, sub in summary.groupby("method"):
        table = to_paper_table(sub)
        path = os.path.join(args.outdir, f"realdata_table_{method}.csv")
        table.to_csv(path)
        print(f"\n=== {method} : rejection rate ({legend}, alpha={args.alpha}) ===")
        print(table.to_string())
        print(f"saved {path}")


if __name__ == "__main__":
    main()
