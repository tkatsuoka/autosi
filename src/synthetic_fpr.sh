#!/bin/bash
# Type I error (FPR) 実験ランチャ兼ジョブ本体。
#   - 引数つきで実行 -> シード範囲ごとに自分自身を qsub でばらまく
#   - PBS から MYARGS つきで実行 -> python を 1 ジョブ分だけ走らせる
# null (signal=0) でデータを生成し、p 値が一様か (FPR が α 近傍か) を見る。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic_fpr.sh                       # 既定で投入
#   bash src/synthetic_fpr.sh -m lasso -d iid -n 10 -p 1000 -w 96
#
# オプション: -m 手法  -d データ種別  -n 分割数(ジョブ数)
#             -p 1ジョブのp値数  -w ワーカー数  -q キュー

#PBS -N synthetic_fpr
#PBS -j oe

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- ジョブ本体: PBS から MYARGS つきで呼ばれた場合 ---
if [ -n "${MYARGS:-}" ]; then
    cd "$PBS_O_WORKDIR"
    uv run python src/run_experiment.py $MYARGS
    exit 0
fi

# --- ランチャ: シード範囲ごとに自分自身を qsub ---
METHOD=lasso;
DATA_TYPE=iid;
NSPLIT=10;
PER=1000;
WORKERS=96;
QUEUE=cpu-medium

while getopts "m:d:n:p:w:q:" o; do case $o in
    m) METHOD=$OPTARG;; d) DATA_TYPE=$OPTARG;; n) NSPLIT=$OPTARG;;
    p) PER=$OPTARG;; w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
esac; done

OUTDIR=result/fpr/${METHOD}_${DATA_TYPE}
mkdir -p "$OUTDIR"

for ((i = 0; i < NSPLIT; i++)); do
    s=$((i * PER)); e=$(((i + 1) * PER))
    args="--methods $METHOD --data_types $DATA_TYPE --signal 0.0 \
          --seed_start $s --seed_end $e --workers $WORKERS \
          --out $OUTDIR/part_${i}.csv"
    qsub -q "$QUEUE" -l select=1:ncpus=${WORKERS}:mem=96gb -l walltime=24:00:00 \
         -v MYARGS="$args" "$0"
done

echo "投入完了: $NSPLIT ジョブ, 計 $((NSPLIT * PER)) 個の p 値 -> $OUTDIR"
echo "完了後: uv run python src/merge_results.py --indir $OUTDIR --out result/fpr_${METHOD}_${DATA_TYPE}.csv"
