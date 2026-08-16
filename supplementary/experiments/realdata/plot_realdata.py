"""Aggregate the real-data results into the table of the paper.

For each (method, subsample size n, dataset), the rejection rate
P(p < alpha) is reported as "proposed / w/o-pp / bonferroni" (the power
proxy on real data; naive does not control the type I error rate and is
excluded from the table, but its rejection rate is kept in
realdata_summary.csv).

Usage (a single CSV file or a directory; directories are read recursively):
  python experiments/realdata/plot_realdata.py --csv result/realdata
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common.merge_results import load_results, limit_per_group


def summarize(df, alpha):
    """Rejection rates per (method, n, dataset) in long format."""
    return (
        df.groupby(["method", "n_samples", "dataset"])
        .agg(
            trials=("p_value", "size"),
            proposed=("p_value", lambda s: (s < alpha).mean()),
            oc=("oc_p_value", lambda s: (s < alpha).mean()),
            naive=("naive_p_value", lambda s: (s < alpha).mean()),
            bonferroni=("bonferroni_p_value", lambda s: (s < alpha).mean()),
        )
        .reset_index()
    )


# Cell layout of the table (naive is invalid and left out)
TABLE_LEGEND = "proposed/w/o-pp/bonferroni"


def to_paper_table(sub):
    """One method: rows = n, columns = dataset, cells = rejection rates."""

    def fmt(r):
        return f"{r.proposed:.2f}/{r.oc:.2f}/{r.bonferroni:.2f}"

    cell = sub.assign(cell=sub.apply(fmt, axis=1))
    return cell.pivot(index="n_samples", columns="dataset", values="cell")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/realdata",
                        help="CSV file or a directory containing CSVs")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="number of p-values used per condition (first "
                             "ones in seed order; default: all)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # Keep the first n_pvalues per (method, n, dataset) in seed order
    df = limit_per_group(df, ["method", "n_samples", "dataset"], args.n_pvalues)

    summary = summarize(df, args.alpha)
    long_path = os.path.join(args.outdir, "realdata_summary.csv")
    summary.to_csv(long_path, index=False)
    print(f"saved {long_path}")

    # One paper-style table per method
    for method, sub in summary.groupby("method"):
        table = to_paper_table(sub)
        path = os.path.join(args.outdir, f"realdata_table_{method}.csv")
        table.to_csv(path)
        print(f"\n=== {method} : rejection rate ({TABLE_LEGEND}, alpha={args.alpha}) ===")
        print(table.to_string())
        print(f"saved {path}")


if __name__ == "__main__":
    main()
