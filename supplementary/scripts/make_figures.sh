#!/bin/bash
# Generate all figures and tables from the result CSVs.
#
# Run from the supplementary root after the experiments have finished:
#   bash scripts/make_figures.sh
set -eu

# Figure 2: type I error rates (+ shared legend legend_fpr.pdf)
python experiments/synthetic/plot_fpr.py --csv result/fpr --outdir result

# Figure 3: power (conditional_power_*.pdf; + shared legend legend_power.pdf)
python experiments/synthetic/plot_power.py --csv result/power --outdir result

# Tables 2 and 4: rejection rates on real data
python experiments/realdata/plot_realdata.py --csv result/realdata --outdir result
