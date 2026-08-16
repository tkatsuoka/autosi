"""Shared plot style for the synthetic-data figures.

Constants keeping figure size, fonts, and legend appearance consistent
between plot_fpr and plot_power, plus a helper that renders a legend-only
PDF shared by all figures (the individual panels carry no legend).
"""

import numpy as np
import matplotlib.pyplot as plt

# Golden-ratio figure size (4-inch width)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
FIGSIZE = (4, 4 / GOLDEN_RATIO)
DPI = 100
FONT_SIZE = 12

# Legend labels. Drawing order = color-cycle order (C0, C1, ...) and must
# match plot_fpr / plot_power. naive is last (C4), so a power legend without
# naive keeps the same colors for the first four methods as the FPR legend.
LABELS = ["proposed", "w/o-pp", "bonferroni", "ds", "naive"]
# Per-method markers (same order as LABELS). Color and marker doubly
# distinguish the curves, robust to overlap, grayscale printing, and
# color-vision deficiency.
MARKERS = ["o", "s", "^", "D", "x"]


def line_style(label):
    """Marker and color for a method label (shared by figures and legend)."""
    i = LABELS.index(label)
    return {"marker": MARKERS[i], "color": f"C{i}", "linestyle": "-"}


def save_legend(path, labels=None):
    """Render a legend-only PDF with the methods in a single row.

    Without `labels`, all methods (LABELS) are drawn. To render a legend for
    a subset (e.g., the power figures), pass a prefix of LABELS so the color
    mapping stays consistent.
    """
    labels = LABELS if labels is None else labels
    plt.rcParams["font.size"] = FONT_SIZE
    fig = plt.figure(figsize=(10, 0.5), dpi=DPI)
    handles = [
        plt.Line2D([], [], label=label, **line_style(label))
        for label in labels
    ]
    fig.legend(handles=handles, ncol=len(labels), loc="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {path}")
