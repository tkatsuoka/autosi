#!/bin/bash
# Real-data experiments on the seven UCI datasets (Tables 2 and 4 of the
# paper): rejection rates over repeated subsamples of size n = 100.
#
# The dynamic mode guarantees 100 p-values per (method, dataset) by
# automatically replacing failed trials (e.g., empty selected set).
#
# Run from the supplementary root:
#   bash scripts/run_realdata.sh [n_pvalues] [workers]
set -eu

N_PVALUES=${1:-100}    # number of p-values per (method, dataset)
WORKERS=${2:-4}        # parallel worker processes

# Download and cache the UCI datasets (skipped when already cached)
python experiments/realdata/datasets.py

for method in marginal_screening lasso cv_r2_lasso; do
  python experiments/realdata/run_realdata.py \
    --methods "$method" \
    --n_samples 100 \
    --target_pvalues "$N_PVALUES" --workers "$WORKERS" \
    --out "result/realdata/${method}.csv"
done
