"""Plot the power experiments (one line chart per condition).

x-axis: signal magnitude Delta; y-axis: power (= fraction of p < alpha).
Two PDFs per (method, data_type) pair:
  conditional_power_<method>_<data_type>.pdf  conditional power
                                              (= P(reject | signal feature selected))
  marginal_power_<method>_<data_type>.pdf     marginal power
                                              (= P(signal feature selected and rejected))
Three curves are drawn (proposed / w/o-pp / bonferroni; colors match the
FPR plots). naive (does not control the FPR) and data splitting (does not
control the type I error under corr) are excluded from the power figures.

Usage (a single CSV file or a directory; directories are read recursively):
  python experiments/synthetic/plot_power.py --csv result/power
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.common.merge_results import load_results, limit_per_group
from experiments.common.plot_style import FIGSIZE, DPI, FONT_SIZE, line_style, save_legend


def _power_curve(sub, p_col, flag_col, alpha, n_pvalues, label, method, data_type):
    """Aggregate one power curve (Delta -> rejection rate).

    Power is measured only over trials where a truly active feature was
    drawn; rejecting a null feature is a type I error and is excluded from
    both the numerator and the denominator (conditional power).
    """
    if flag_col in sub.columns:
        n_total = len(sub)
        sub = sub[sub[flag_col]]
        # Rows whose SI inference was skipped after the target count was
        # reached in dynamic mode have empty p-values; drop them (this
        # truncation follows seed order and does not depend on the p-values,
        # so no bias is introduced).
        sub = sub[sub[p_col].notna()]
        print(f"{method} ({data_type}) [{label}]: computing power over "
              f"{len(sub)}/{n_total} valid p-values")
    else:
        print(f"warning: {method} ({data_type}) lacks the {flag_col} column; "
              "computing power over all trials.")

    # After the signal filter, keep the first n_pvalues per Delta
    sub = limit_per_group(sub, ["signal"], n_pvalues)

    # Power per Delta = P(p < alpha | signal feature selected)
    return (
        sub.assign(reject=sub[p_col] < alpha)
        .groupby("signal")["reject"].mean()
        .sort_index()
    )


def _selection_rate(sub, flag_col):
    """Selection success rate P(signal feature selected) per Delta,
    estimated over all trials."""
    return sub.groupby("signal")[flag_col].mean().sort_index()


def plot_one(sub, method, data_type, alpha, outdir, n_pvalues=None,
             unconditional=False):
    """Draw the power line chart for one (method, data_type) pair.

    With unconditional=True the marginal power
    = P(signal feature selected and rejected)
    = selection success rate x conditional power
    is drawn instead. The selection rate is estimated over all trials
    (--n_pvalues applies only to the conditional part).
    """
    # (legend label, p-value column, signal flag column, zorder); order
    # shared with plot_fpr.
    curves = [
        ("proposed", "p_value", "is_signal", 10),
        ("w/o-pp", "oc_p_value", "is_signal", 9),
        ("bonferroni", "bonferroni_p_value", "is_signal", 8),
    ]

    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    sigs = sorted(sub["signal"].unique())
    ax.plot(sigs, [alpha] * len(sigs), linestyle="dashed", color="black")
    for label, p_col, flag_col, z in curves:
        if p_col not in sub.columns:
            continue
        power = _power_curve(sub, p_col, flag_col, alpha, n_pvalues,
                             label, method, data_type)
        if unconditional:
            if flag_col not in sub.columns:
                continue
            # Marginal power = selection rate x conditional power
            power = power * _selection_rate(sub, flag_col)
        ax.plot(power.index, power.values, label=label, zorder=z,
                **line_style(label))

    ax.set_xticks(sigs)
    # Major ticks every 0.2 + unlabeled minor ticks every 0.1 (as in FPR)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1), minor=True)
    ax.set_xlabel("Signal")
    # Same y-label for conditional/marginal; the file name distinguishes them
    ax.set_ylabel("Power")
    prefix = "marginal_power" if unconditional else "conditional_power"
    path = os.path.join(outdir, f"{prefix}_{method}_{data_type}.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="result/power",
                        help="CSV file or a directory containing CSVs")
    parser.add_argument("--outdir", type=str, default="result")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_pvalues", type=int, default=None,
                        help="number of p-values used per condition (after "
                             "the is_signal filter, first ones in seed "
                             "order; default: all)")
    args = parser.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # One conditional and one marginal figure per (method, data_type)
    for (method, data_type), sub in df.groupby(["method", "data_type"]):
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues)
        plot_one(sub, method, data_type, args.alpha, args.outdir,
                 n_pvalues=args.n_pvalues, unconditional=True)

    # Shared legend-only PDF for the power figures
    save_legend(os.path.join(args.outdir, "legend_power.pdf"),
                labels=["proposed", "w/o-pp", "bonferroni"])


if __name__ == "__main__":
    main()
