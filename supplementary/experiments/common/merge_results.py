"""Loading and merging result CSV files.

Usage (merging split runs into one file is optional; the plotting scripts
accept a directory directly):
  python experiments/common/merge_results.py --indir result/parts --out result/merged.csv
"""

import os
import glob
import argparse

import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Load a single CSV file, or recursively concatenate all CSVs under a
    directory."""
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "**", "*.csv"),
                                 recursive=True))
        if not paths:
            raise SystemExit(f"no csv found under {path}")
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return pd.read_csv(path)


def limit_per_group(df: pd.DataFrame, group_cols: list[str],
                    n_pvalues: int | None, sort_col: str = "seed") -> pd.DataFrame:
    """Keep the first n_pvalues rows (in sort_col order) per condition.

    Used to equalize the number of p-values across conditions when more
    were computed as a margin against failed trials. Returns df unchanged
    when n_pvalues is None; warns about conditions that fall short.
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
        print(f"warning: the following conditions have fewer than "
              f"{n_pvalues} p-values (using what is available):")
        print(short.to_string())
    return limited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="result/parts")
    parser.add_argument("--out", type=str, default="result/merged.csv")
    args = parser.parse_args()

    df = load_results(args.indir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"merged {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
