#!/bin/bash
# 実データ実験ドライバ。
# UCI の 8 データセットを n 行サブサンプリングして p 値を計算する
# (si4pipeline 論文の実データ実験)。(データセット × n × 分割) ごとに
# PBS ジョブを 1 本ずつ投入する (ジョブ本体は src/jobs/realdata.pbs)。
#
# 注意: データセットの取得にはネットワークが必要なため、ドライバ側 (ログイン
# ノード) で先にダウンロードし、ジョブはキャッシュ済み CSV のみ読む。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/realdata/realdata.sh                        # 既定で投入
#   bash src/realdata/realdata.sh -m lasso -D "wine_red wine_white" -N "100 150 200"
#
# オプション: -m 手法  -D "データセットのリスト"  -N "サンプル数のリスト"
#             -n 分割数((データセット,n)ごとのジョブ数)  -p 1ジョブのp値数
#             -w ワーカー数  -q キュー

set -euo pipefail
source "$(dirname "$0")/../pbs_common.sh"

METHOD=lasso
DATASETS="airfoil concrete energy_heating energy_cooling gas_turbine real_estate wine_red wine_white"
NLIST="100 150 200"          # サブサンプルのサイズ n のリスト (論文と同じ)
NSPLIT=10                    # (データセット, n) ごとのジョブ数 (= 分割数)
PER=110                      # 1 ジョブが計算する p 値の個数 (10 分割 × 110 = 1100、エラー時の不足に備え 1 割の余裕)

while getopts "m:D:N:n:p:w:q:" o; do case $o in
    m) METHOD=$OPTARG;; D) DATASETS=$OPTARG;; N) NLIST=$OPTARG;;
    n) NSPLIT=$OPTARG;; p) PER=$OPTARG;; w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
esac; done

# 手法ごとに seed をずらし、手法間で異なるサブサンプル (独立) にする
case $METHOD in
    marginal_screening) OFFSET=0;;
    lasso)              OFFSET=1000000;;
    cv_r2_lasso)        OFFSET=2000000;;
    *)                  OFFSET=0;;
esac

# データセットをログインノードで先に取得する (計算ノードはキャッシュ参照のみ)
echo "データセットを準備しています (キャッシュ済みならスキップ)..."
"$PYTHON" src/realdata/datasets.py

count=0
for ds in $DATASETS; do
    for ns in $NLIST; do
        OUTDIR="result/realdata/${METHOD}/${ds}_n${ns}"
        mkdir -p "$OUTDIR"
        for i in $(seq 0 $((NSPLIT - 1))); do
            s=$((i * PER)); e=$(((i + 1) * PER))
            MYARGS="--methods $METHOD --datasets $ds --n_samples $ns"
            MYARGS+=" --seed_start $s --seed_end $e --seed_offset $OFFSET"
            MYARGS+=" --workers $WORKERS --out $OUTDIR/part_${i}.csv"
            submit_job "rd_${METHOD}_${ds}_n${ns}_${i}" "$REALDATA_JOB" "$MYARGS"
            count=$((count + 1))
        done
    done
done

echo "投入完了: ${count} ジョブ -> result/realdata/${METHOD}/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/realdata/plot_realdata.py --csv result/realdata"
