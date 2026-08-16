#!/bin/bash
# Type I error rate experiments on synthetic data (Figure 2 of the paper).
#
# For every method x data type x sample size, 1000 null p-values are
# collected (signal = 0). Run from the supplementary root:
#   bash scripts/run_synthetic_fpr.sh [n_pvalues] [workers]
#
# The full setting is computationally heavy (especially cv_r2_lasso); for a
# quick check use e.g. `bash scripts/run_synthetic_fpr.sh 10 4`. To split
# the work across machines, run subsets with --seed_start/--seed_end and
# write to separate part files (the plot script reads a directory).
set -eu

N_PVALUES=${1:-1000}   # number of null p-values per condition
WORKERS=${2:-4}        # parallel worker processes

for method in marginal_screening lasso cv_r2_lasso; do
  for data_type in iid corr; do
    for n in 100 200 300 400; do
      python experiments/synthetic/run_experiment.py \
        --methods "$method" --data_types "$data_type" \
        --n_samples "$n" --signal 0.0 \
        --iter "$N_PVALUES" --workers "$WORKERS" \
        --out "result/fpr/${method}_${data_type}_n${n}.csv"
    done
  done
done
