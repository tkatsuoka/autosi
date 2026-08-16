"""Plot the type I error rate experiments (one line chart per condition).

x-axis: sample size n; y-axis: type I error rate (= fraction of p < alpha).
One PDF per (method, data_type) pair: result/fpr_<method>_<data_type>.pdf.

Usage (a single CSV file or a directory; directories are read recursively):
  python experiments/synthetic/plot_fpr.py --csv result/fpr
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common.merge_results import load_results, limit_per_group
from experiments.common.plot_style import FIGSIZE, DPI, FONT_SIZE, line_style, save_legend


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None):
    """Draw the FPR line chart for one (method, data_type) pair."""
    # Keep the first n_pvalues per sample size (in seed order)
    sub = limit_per_group(sub, ["n_samples"], n_pvalues)
    # (legend label, p-value column, zorder); drawing order = legend order =
    # color-cycle order, with the proposed method in front (shared with
    # plot_power).
    curves = [
        ("proposed", "p_value", 10),
        ("w/o-pp", "oc_p_value", 9),
        ("bonferroni", "bonferroni_p_value", 8),
        ("ds", "ds_p_value", 7),
        ("naive", "naive_p_value", 6),
    ]

    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    ns = sorted(sub["n_samples"].unique())
    ax.plot(ns, [alpha] * len(ns), linestyle="dashed", color="black")
    for label, col, z in curves:
        if col not in sub.columns:
            continue
        # Type I error rate = P(p < alpha) per sample size
        rate = (
            sub.assign(reject=sub[col] < alpha)
            .groupby("n_samples")["reject"].mean()
            .sort_index()
        )
        ax.plot(rate.index, rate.values, label=label, zorder=z,
                **line_style(label))

    ax.set_xticks(ns)
    # Major ticks every 0.2 (aligned with the power figures), unlabeled
    # minor ticks every 0.1
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1), minor=True)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Type I Error Rate")
    # No per-panel legend; the shared legend PDF is rendered separately
    path = os.path.join(outdir, f"fpr_{method}_{data_type}.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/fpr",
                        help="CSV file or a directory containing CSVs")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="number of p-values used per condition (first "
                             "ones in seed order; default: all)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    for (method, data_type), sub in df.groupby(["method", "data_type"]):
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues)

    # Shared legend-only PDF for the FPR figures (all 5 methods)
    save_legend(os.path.join(args.outdir, "legend_fpr.pdf"))


if __name__ == "__main__":
    main()
