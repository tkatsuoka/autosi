#!/bin/bash
# 検出力 (power) 実験ドライバ。
# y = Xβ + ε (先頭 N_SIGNAL 個の係数を Δ) でデータを生成し、(Δ × 分割) ごとに
# PBS ジョブを 1 本ずつ投入する (ジョブ本体は src/jobs/synthetic.pbs)。
# n は run_experiment.py の既定 (200) に固定される。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/power.sh                                  # 既定で投入
#   bash src/synthetic/power.sh -m lasso -d iid -n 10 -p 1100 -s "0.1 0.2 0.3 0.4"
#
# オプション: -m 手法  -d データ種別 (iid / corr)  -n 分割数(Δごとのジョブ数)
#             -p 1ジョブのp値数  -s "Δのリスト"  -k 信号特徴量数
#             -w ワーカー数  -q キュー

set -euo pipefail
source "$(dirname "$0")/../pbs_common.sh"

METHOD=lasso
DATA_TYPE=iid
NSPLIT=10                    # Δ ごとのジョブ数 (= 分割数)
PER=1100                     # 1 ジョブが計算する p 値の個数 (エラー時の不足に備え 1 割の余裕)
SIGNALS="0.1 0.2 0.3 0.4"    # 横軸となる信号 Δ のリスト
N_SIGNAL=5

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

count=0
for sig in $SIGNALS; do
    OUTDIR="result/power/${METHOD}_${DATA_TYPE}_d${sig}"
    mkdir -p "$OUTDIR"
    for i in $(seq 0 $((NSPLIT - 1))); do
        s=$((i * PER)); e=$(((i + 1) * PER))
        MYARGS="--methods $METHOD --data_types $DATA_TYPE"
        MYARGS+=" --signal $sig --n_signal $N_SIGNAL"
        MYARGS+=" --seed_start $s --seed_end $e --seed_offset $OFFSET"
        MYARGS+=" --workers $WORKERS --out $OUTDIR/part_${i}.csv"
        # ジョブ名に '.' は使えない場合があるため Δ からは除く (0.1 -> d01)
        submit_job "power_${METHOD}_${DATA_TYPE}_d${sig//./}_${i}" "$SYNTHETIC_JOB" "$MYARGS"
        count=$((count + 1))
    done
done

echo "投入完了: ${count} ジョブ -> result/power/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/synthetic/plot_power.py --csv result/power"
