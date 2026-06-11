#!/bin/bash
# 検出力 (power) 実験用ランチャ。
# 信号の傾き Δ を振り、「手法 × データ種別 × Δ」ごとに 1 ジョブを投入する。
# power = (p 値 < α の割合) を Δ ごとに見ることで、信号が強いほど検出力が
# 上がる曲線を描く。論文流に先頭 N_SIGNAL 個の係数を Δ にする。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/submit_power.sh

cd "$PBS_O_WORKDIR" 2>/dev/null || true

# ===== 実験条件 =====
ITER=200            # 各 (手法,データ種別,Δ) で計算する p 値の個数
SIGNALS="0.2 0.4 0.6 0.8"   # 振る Δ
N_SIGNAL=3          # 信号を載せる先頭特徴量の数

# ===== 並列・キュー設定 =====
WORKERS=96
QUEUE="cpu-medium"
NCPUS=${WORKERS}
MEM="96gb"
WALLTIME="24:00:00"
OUTDIR="result/power_parts"
mkdir -p "$OUTDIR"

METHODS="marginal_screening lasso cv_r2_lasso"
DATA_TYPES="iid correlated"

for method in $METHODS; do
    for dt in $DATA_TYPES; do
        for sig in $SIGNALS; do
            out="${OUTDIR}/${method}_${dt}_d${sig}.csv"
            args="--methods ${method} --data_types ${dt} \
                  --iter ${ITER} --signal ${sig} --n_signal ${N_SIGNAL} \
                  --workers ${WORKERS} --out ${out}"
            echo "submit: $method / $dt / Δ=$sig -> $out"
            qsub -q "${QUEUE}" \
                 -l select=1:ncpus=${NCPUS}:mem=${MEM} \
                 -l walltime=${WALLTIME} \
                 -v MYARGS="${args}" src/job.sh
        done
    done
done

echo "投入完了。全ジョブ終了後に以下で統合・作図してください:"
echo "  uv run python src/merge_results.py --indir ${OUTDIR} --out result/power_results.csv"
echo "  uv run python src/plot_power.py --csv result/power_results.csv"
