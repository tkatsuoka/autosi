#!/bin/bash
# Type I error (FPR) 実験ドライバ。
# null (signal=0) データでサンプル数 n を振り、(n × 分割) ごとに
# PBS ジョブを 1 本ずつ投入する (ジョブ本体は src/jobs/synthetic.pbs)。
#
# power.sh と同じ動的モード: 各ジョブは有効 p 値が -p 本たまるまで試行を
# 自動追加する。null では選択が空になった試行 (大きい n で lasso 系に発生)
# がスキップされるため、固定回数だと本数が目減りするのを防ぐ。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/fpr.sh                       # 既定で投入
#   bash src/synthetic/fpr.sh -m lasso -d iid -N "100 200 300 400" -n 10 -p 110
#
# オプション: -m 手法  -d データ種別 (iid / corr)  -N "サンプル数のリスト"
#             -n 分割数(サンプル数ごとのジョブ数)  -p 1ジョブの有効p値数
#             -w ワーカー数  -q キュー

set -euo pipefail
source "$(dirname "$0")/../pbs_common.sh"

METHOD=lasso
DATA_TYPE=iid
NSAMPLES="100 200 300 400"   # 横軸となるサンプル数のリスト
NSPLIT=10                    # サンプル数ごとのジョブ数 (= 分割数)
PER=110                      # 1 ジョブが集める有効 p 値の本数 (10 分割 × 110 = 1100 本/条件)
SEED_STRIDE=10000            # part ごとの seed ブロック幅 (動的モードは試行数が変動するため重複防止)

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

count=0
for ns in $NSAMPLES; do
    OUTDIR="result/fpr/${METHOD}_${DATA_TYPE}/n${ns}"
    mkdir -p "$OUTDIR"
    for i in $(seq 0 $((NSPLIT - 1))); do
        # 動的モード: 各 part は有効 p 値 PER 本を集めるまで試行を追加する。
        # 消費する seed 数が変動するため SEED_STRIDE 刻みのブロックで重複を防ぐ
        MYARGS="--methods $METHOD --data_types $DATA_TYPE --signal 0.0"
        MYARGS+=" --n_samples $ns --target_pvalues $PER"
        MYARGS+=" --seed_start $((i * SEED_STRIDE)) --seed_offset $OFFSET"
        MYARGS+=" --workers $WORKERS --out $OUTDIR/part_${i}.csv"
        submit_job "fpr_${METHOD}_${DATA_TYPE}_n${ns}_${i}" "$SYNTHETIC_JOB" "$MYARGS"
        count=$((count + 1))
    done
done

echo "投入完了: ${count} ジョブ -> result/fpr/${METHOD}_${DATA_TYPE}/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/synthetic/plot_fpr.py --csv result/fpr/${METHOD}_${DATA_TYPE}"
