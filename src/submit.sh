#!/bin/bash
# 実験を「手法 × データ種別」ごとに分割し、qsub で並列投入するランチャ。
# src/ref/experiment_iid.sh と同じく、ループで条件を展開して 1 条件 = 1 ジョブを投入する。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/submit.sh
#
# 各ジョブの結果は result/parts/<method>_<data_type>.csv に出力され、
# 全ジョブ完了後に merge_results.py で 1 つの CSV に統合する。

cd "$PBS_O_WORKDIR" 2>/dev/null || true

# ===== 実験条件 =====
ITER=200          # 各条件で計算する p 値の個数 (= seed 0..ITER-1 のシード数)
SIGNAL=0.0        # 真の係数の傾き。0.0: null (FPR/一様性), >0: power

# ===== 並列・キュー設定 =====
WORKERS=96        # 1 ジョブ内の並列ワーカー数 (python の --workers)
QUEUE="cpu-medium"
NCPUS=${WORKERS}  # 確保するコア数。ワーカー数と揃える
MEM="96gb"
WALLTIME="24:00:00"
OUTDIR="result/parts"
mkdir -p "$OUTDIR"

METHODS="marginal_screening lasso cv_r2_lasso"
DATA_TYPES="iid correlated"

for method in $METHODS; do
    for dt in $DATA_TYPES; do
        out="${OUTDIR}/${method}_${dt}.csv"
        args="--methods ${method} --data_types ${dt} \
              --iter ${ITER} --signal ${SIGNAL} \
              --workers ${WORKERS} --out ${out}"
        echo "submit: $method / $dt -> $out"
        qsub -q "${QUEUE}" \
             -l select=1:ncpus=${NCPUS}:mem=${MEM} \
             -l walltime=${WALLTIME} \
             -v MYARGS="${args}" src/job.sh
    done
done

echo "投入完了。全ジョブ終了後に以下で統合してください:"
echo "  uv run python src/merge_results.py --indir ${OUTDIR} --out result/experiment_results.csv"
