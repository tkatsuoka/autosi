#!/bin/bash
# Type I error (FPR) 実験ランチャ兼ジョブ本体。
#   - 引数つきで実行 -> シード範囲ごとに自分自身を qsub でばらまく
#   - PBS から MYARGS つきで実行 -> python を 1 ジョブ分だけ走らせる
# null (signal=0) でデータを生成し、p 値が一様か (FPR が α 近傍か) を見る。
#
# サンプル数 n を振り (横軸)、各 n で type I error rate を測る。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic_fpr.sh                       # 既定で投入
#   bash src/synthetic_fpr.sh -m lasso -d iid -N "100 200 300 400" -n 10 -p 1000
#
# オプション: -m 手法  -d データ種別  -N "サンプル数のリスト"
#             -n 分割数(ジョブ数)  -p 1ジョブのp値数  -w ワーカー数  -q キュー

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

# --- ランチャ: サンプル数 n × シード範囲ごとに自分自身を qsub ---
METHOD=lasso
DATA_TYPE=iid
NSAMPLES="100 200 300 400"   # 横軸となるサンプル数のリスト
NSPLIT=10
PER=1000
WORKERS=96
QUEUE=cpu-medium

while getopts "m:d:N:n:p:w:q:" o; do case $o in
    m) METHOD=$OPTARG;; d) DATA_TYPE=$OPTARG;; N) NSAMPLES=$OPTARG;;
    n) NSPLIT=$OPTARG;; p) PER=$OPTARG;; w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
esac; done

# 手法ごとに seed をずらし、手法間で異なるデータ (独立) にする
case $METHOD in
    marginal_screening) OFFSET=0;;
    lasso)              OFFSET=1000000;;
    cv_r2_lasso)        OFFSET=2000000;;
    *)                  OFFSET=0;;
esac

for ns in $NSAMPLES; do
    OUTDIR=result/fpr/${METHOD}_${DATA_TYPE}/n${ns}
    mkdir -p "$OUTDIR"
    for ((i = 0; i < NSPLIT; i++)); do
        s=$((i * PER)); e=$(((i + 1) * PER))
        args="--methods $METHOD --data_types $DATA_TYPE --signal 0.0 \
              --n_samples $ns --seed_start $s --seed_end $e --seed_offset $OFFSET \
              --workers $WORKERS --out $OUTDIR/part_${i}.csv"
        qsub -q "$QUEUE" -l select=1:ncpus=${WORKERS}:mem=96gb -l walltime=24:00:00 \
             -v MYARGS="$args" "$0"
    done
done

echo "投入完了 -> result/fpr/${METHOD}_${DATA_TYPE}/ 以下。完了後:"
echo "  uv run python src/plot_results.py --csv result/fpr/${METHOD}_${DATA_TYPE}"
