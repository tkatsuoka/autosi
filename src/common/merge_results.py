"""qsub で分割実行した複数の CSV を 1 つに統合する。

使い方:
  python src/merge_results.py --indir result/parts --out result/experiment_results.csv
"""

import os
import glob
import argparse

import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """CSV ファイル単体、またはフォルダ (再帰的に全 CSV) を読み込んで結合する。

    描画スクリプトから「統合せずに直接フォルダを渡す」用途にも使う。
    """
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "**", "*.csv"),
                                 recursive=True))
        if not paths:
            raise SystemExit(f"no csv found under {path}")
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return pd.read_csv(path)


def limit_per_group(df: pd.DataFrame, group_cols: list[str],
                    n_pvalues: int | None, sort_col: str = "seed") -> pd.DataFrame:
    """条件 (group_cols) ごとに sort_col 順で先頭 n_pvalues 行だけ残す。

    エラー時の不足に備えて多めに計算した p 値から、プロットに使う個数を
    条件間で揃えるために使う。n_pvalues が None なら全件をそのまま返す。
    不足している条件があれば警告を表示する。
    """
    if n_pvalues is None:
        return df
    limited = (
        df.sort_values(sort_col)
        .groupby(group_cols, group_keys=False)
        .head(n_pvalues)
    )
    counts = limited.groupby(group_cols).size()
    short = counts[counts < n_pvalues]
    if not short.empty:
        print(f"警告: 以下の条件は p 値が {n_pvalues} 件に足りません "
              "(ある分だけで集計します):")
        print(short.to_string())
    return limited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="result/parts")
    parser.add_argument("--out", type=str, default="result/experiment_results.csv")
    args = parser.parse_args()

    df = load_results(args.indir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"merged {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
