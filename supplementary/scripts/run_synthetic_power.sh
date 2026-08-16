#!/bin/bash
# Power experiments on synthetic data (Figure 3 of the paper).
#
# n = 100 is fixed and the signal magnitude Delta is varied. Because power
# is computed only over trials whose selection picked a truly active
# feature, the runner uses the dynamic mode (--target_pvalues): trials are
# added until 1000 valid p-values are collected per condition.
#
# Run from the supplementary root:
#   bash scripts/run_synthetic_power.sh [n_pvalues] [workers]
set -eu

N_PVALUES=${1:-1000}   # number of valid p-values per condition
WORKERS=${2:-4}        # parallel worker processes

for method in marginal_screening lasso cv_r2_lasso; do
  for data_type in iid corr; do
    for delta in 0.1 0.2 0.3 0.4; do
      python experiments/synthetic/run_experiment.py \
        --methods "$method" --data_types "$data_type" \
        --n_samples 100 --signal "$delta" \
        --target_pvalues "$N_PVALUES" --workers "$WORKERS" \
        --out "result/power/${method}_${data_type}_d${delta}.csv"
    done
  done
done
