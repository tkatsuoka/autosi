#!/bin/bash
# 検出力 (power) 実験ランチャ兼ジョブ本体。
#   - 引数つきで実行 -> 信号Δ × シード範囲ごとに自分自身を qsub でばらまく
#   - PBS から MYARGS つきで実行 -> python を 1 ジョブ分だけ走らせる
# y = Xβ + ε (先頭 N_SIGNAL 個の係数を Δ) でデータを生成し、
# power = P(p 値 < α) を Δ ごとに見る。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic_power.sh                                  # 既定で投入
#   bash src/synthetic_power.sh -m lasso -d iid -n 10 -p 1000 -s "0.1 0.2 0.3 0.4"
#
# オプション: -m 手法  -d データ種別  -n 分割数(ジョブ数)  -p 1ジョブのp値数
#             -s "Δのリスト"  -k 信号特徴量数  -w ワーカー数  -q キュー

#PBS -N synthetic_power
#PBS -j oe

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- ジョブ本体: PBS から MYARGS つきで呼ばれた場合 ---
if [ -n "${MYARGS:-}" ]; then
    cd "$PBS_O_WORKDIR"
    uv run python src/run_experiment.py $MYARGS
    exit 0
fi

# --- ランチャ: 信号Δ × シード範囲ごとに自分自身を qsub ---
METHOD=lasso;
DATA_TYPE=iid;
NSPLIT=10;
PER=1000;
SIGNALS="0.1 0.2 0.3 0.4";
N_SIGNAL=5;
WORKERS=96;
QUEUE=cpu-medium

while getopts "m:d:n:p:s:k:w:q:" o; do case $o in
    m) METHOD=$OPTARG;; d) DATA_TYPE=$OPTARG;; n) NSPLIT=$OPTARG;;
    p) PER=$OPTARG;; s) SIGNALS=$OPTARG;; k) N_SIGNAL=$OPTARG;;
    w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
esac; done

# 手法ごとに seed をずらし、手法間で異なるデータ (独立) にする
case $METHOD in
    marginal_screening) OFFSET=0;;
    lasso)              OFFSET=1000000;;
    cv_r2_lasso)        OFFSET=2000000;;
    *)                  OFFSET=0;;
esac

for sig in $SIGNALS; do
    OUTDIR=result/power/${METHOD}_${DATA_TYPE}_d${sig}
    mkdir -p "$OUTDIR"
    for ((i = 0; i < NSPLIT; i++)); do
        s=$((i * PER)); e=$(((i + 1) * PER))
        args="--methods $METHOD --data_types $DATA_TYPE \
              --signal $sig --n_signal $N_SIGNAL \
              --seed_start $s --seed_end $e --seed_offset $OFFSET --workers $WORKERS \
              --out $OUTDIR/part_${i}.csv"
        qsub -q "$QUEUE" -l select=1:ncpus=${WORKERS}:mem=256gb -l walltime=72:00:00 \
             -v MYARGS="$args" "$0"
    done
done

echo "投入完了 -> result/power/ 以下。完了後:"
echo "  uv run python src/merge_results.py --indir result/power --out result/power_results.csv"
echo "  uv run python src/plot_power.py --csv result/power_results.csv"
