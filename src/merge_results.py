"""qsub で分割実行した複数の CSV を 1 つに統合する。

使い方:
  python src/merge_results.py --indir result/parts --out result/experiment_results.csv
"""

import os
import glob
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="result/parts")
    parser.add_argument("--out", type=str, default="result/experiment_results.csv")
    args = parser.parse_args()

    # 直下とサブフォルダ (power はΔごとにサブフォルダ) を再帰的に拾う
    paths = sorted(glob.glob(os.path.join(args.indir, "**", "*.csv"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no csv found under {args.indir}")

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"merged {len(paths)} files, {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
