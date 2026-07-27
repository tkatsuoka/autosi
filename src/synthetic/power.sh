#!/bin/bash
# 検出力 (power) 実験ドライバ。
# y = Xβ + ε (先頭 N_SIGNAL 個の係数を Δ) でデータを生成し、(Δ × 分割) ごとに
# PBS ジョブを 1 本ずつ投入する (ジョブ本体は src/jobs/synthetic.pbs)。
# n は既定 200 に固定 (-N で変更可)。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/power.sh                                  # 既定で投入
#   bash src/synthetic/power.sh -m lasso -d iid -n 10 -p 110 -s "0.1 0.2 0.3 0.4"
#
# p 値の本数は動的モード: 各ジョブは「選択に成功した試行の有効 p 値」が
# -p 本たまるまで試行を自動追加する (proposed 系と DS の両基準)。
#
# オプション: -m 手法  -d データ種別 (iid / corr)  -N サンプル数 n
#             -n 分割数(Δごとのジョブ数)  -p 1ジョブの有効p値数
#             -s "Δのリスト"  -k 信号特徴量数  -w ワーカー数  -q キュー

set -euo pipefail
source "$(dirname "$0")/../pbs_common.sh"

METHOD=lasso
DATA_TYPE=iid
NSAMPLES=200                 # サンプル数 n (固定)
NSPLIT=10                    # Δ ごとのジョブ数 (= 分割数)
PER=110                      # 1 ジョブが集める有効 p 値の本数 (is_signal / ds_is_signal 各基準)。
                             # 動的モードで不足分は自動追加されるため 10 分割 × 110 = 1100 本/条件が保証される
SIGNALS="0.1 0.2 0.3 0.4"    # 横軸となる信号 Δ のリスト
N_SIGNAL=3
SEED_STRIDE=10000            # part ごとの seed ブロック幅 (動的モードは試行数が変動するため重複防止)

while getopts "m:d:N:n:p:s:k:w:q:" o; do case $o in
    m) METHOD=$OPTARG;; d) DATA_TYPE=$OPTARG;; N) NSAMPLES=$OPTARG;;
    n) NSPLIT=$OPTARG;; p) PER=$OPTARG;; s) SIGNALS=$OPTARG;;
    k) N_SIGNAL=$OPTARG;; w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
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
        # 動的モード: 各 part は有効 p 値 PER 本を集めるまで試行を追加する。
        # 消費する seed 数が変動するため、part ごとに SEED_STRIDE 刻みの
        # ブロックを割り当てて重複を防ぐ
        MYARGS="--methods $METHOD --data_types $DATA_TYPE"
        MYARGS+=" --n_samples $NSAMPLES --signal $sig --n_signal $N_SIGNAL"
        MYARGS+=" --target_pvalues $PER --seed_start $((i * SEED_STRIDE))"
        MYARGS+=" --seed_offset $OFFSET"
        MYARGS+=" --workers $WORKERS --out $OUTDIR/part_${i}.csv"
        # ジョブ名に '.' は使えない場合があるため Δ からは除く (0.1 -> d01)
        submit_job "power_${METHOD}_${DATA_TYPE}_d${sig//./}_${i}" "$SYNTHETIC_JOB" "$MYARGS"
        count=$((count + 1))
    done
done

echo "投入完了: ${count} ジョブ -> result/power/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/synthetic/plot_power.py --csv result/power"
