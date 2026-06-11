#!/bin/bash
#PBS -N autosi
#PBS -q cpu-medium
#PBS -l select=1:ncpus=96:mem=96gb
#PBS -l walltime=24:00:00
#PBS -j oe

# qsub から MYARGS 経由で渡された引数で 1 ジョブ分の実験を実行する。
#   例: qsub -v MYARGS="--methods lasso --data_types iid --seed_start 0 --seed_end 50 --out result/lasso_iid_0.csv" src/job.sh
#
# 投入元ディレクトリ (qsub を叩いた場所) に移動してから実行する。
cd "$PBS_O_WORKDIR" || exit 1

# BLAS のスレッド過剰生成を防ぐ (並列ワーカーと競合させない)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host: $(hostname)"
echo "args: $MYARGS"

# uv 環境で実行 (uv が無い場合は `python` に置き換える)
uv run python src/run_experiment.py $MYARGS
