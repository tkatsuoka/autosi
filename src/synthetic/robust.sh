#!/bin/bash
# ロバスト性 (type I error) 実験ドライバ。
# null (signal=0) の iid データで「正規ノイズ・既知分散」の仮定を崩したとき
# proposed が type I error を制御できるかを調べる。2 種類の実験を、
# (条件 × 分割) ごとに PBS ジョブ 1 本ずつ投入する (本体は src/jobs/synthetic.pbs)。
#   var   : 分散を同じデータから推定 (σ̂² = RSS/(n-d)) し、サンプル数 n を振る
#   noise : ノイズを非ガウス分布 (5 分布族) に替え、N(0,1) からの
#           1-Wasserstein 距離を振る (n は power 実験と同じ 200 に固定)
# 比較手法は計算せず proposed のみ (--proposed_only)。α ごとの棄却率は
# 同じ p 値から描画時に計算する (plot_robust.py)。
#
# seed は fpr.sh と同じ割り当てなので、var 実験は FPR 実験 (iid) と同じ
# データで「既知分散 vs 推定分散」を比べる対応のある比較になる。
#
# ジョブ数 (既定): var 4 条件 × 10 分割 = 40、noise 20 条件 × 10 分割 = 200 / 手法。
# 重い手法では -n 5 -p 220 のように分割を粗くしてジョブ数を減らせる。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/robust.sh                              # 既定 (lasso, var + noise)
#   bash src/synthetic/robust.sh -m cv_r2_lasso -t noise -n 5 -p 220
#
# オプション: -m 手法  -t "実験のリスト" (var / noise)
#             -N "var のサンプル数のリスト"  -S noise のサンプル数
#             -F "ノイズ分布族のリスト"  -D "Wasserstein 距離のリスト"
#             -n 分割数(条件ごとのジョブ数)  -p 1ジョブの有効p値数
#             -w ワーカー数  -q キュー

set -euo pipefail
source "$(dirname "$0")/../pbs_common.sh"

METHOD=lasso
TYPES="var noise"                                          # 投入する実験
VAR_NSAMPLES="100 200 300 400"                             # var の横軸 (FPR 実験と同じ)
NOISE_NSAMPLES=200                                         # noise の n (power 実験と同じ)
FAMILIES="skewnorm exponnorm gennormsteep gennormflat t"  # 非ガウス分布族
DISTANCES="0.01 0.02 0.03 0.04"                            # noise の横軸
NSPLIT=10                    # 条件ごとのジョブ数 (= 分割数)
PER=110                      # 1 ジョブが集める有効 p 値の本数 (10 分割 × 110 = 1100 本/条件)
SEED_STRIDE=10000            # part ごとの seed ブロック幅 (動的モードは試行数が変動するため重複防止)

while getopts "m:t:N:S:F:D:n:p:w:q:" o; do case $o in
    m) METHOD=$OPTARG;; t) TYPES=$OPTARG;; N) VAR_NSAMPLES=$OPTARG;;
    S) NOISE_NSAMPLES=$OPTARG;; F) FAMILIES=$OPTARG;; D) DISTANCES=$OPTARG;;
    n) NSPLIT=$OPTARG;; p) PER=$OPTARG;; w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
esac; done

# 手法ごとに seed をずらし、手法間で異なるデータ (独立) にする (fpr.sh と同じ)
case $METHOD in
    marginal_screening) OFFSET=0;;
    lasso)              OFFSET=1000000;;
    cv_r2_lasso)        OFFSET=2000000;;
    *)                  OFFSET=0;;
esac

# 途中まで投入してから失敗しないよう、実験名は投入前にまとめて検証する
for t in $TYPES; do
    case $t in
        var|noise) ;;
        *) echo "エラー: 未知の実験 '$t' (指定可能: var / noise)" >&2; exit 1;;
    esac
done

count=0

# 1 条件分の NSPLIT 本のジョブを投入する
#   使い方: submit_parts <ジョブ名> <出力フォルダ> "<条件固有の引数>"
submit_parts() {
    local name="$1" outdir="$2" extra="$3"
    mkdir -p "$outdir"
    for i in $(seq 0 $((NSPLIT - 1))); do
        MYARGS="--methods $METHOD --data_types iid --signal 0.0 --proposed_only $extra"
        MYARGS+=" --target_pvalues $PER --seed_start $((i * SEED_STRIDE))"
        MYARGS+=" --seed_offset $OFFSET"
        MYARGS+=" --workers $WORKERS --out $outdir/part_${i}.csv"
        submit_job "${name}_${i}" "$SYNTHETIC_JOB" "$MYARGS"
        count=$((count + 1))
    done
}

for t in $TYPES; do
    case $t in
        var)
            for ns in $VAR_NSAMPLES; do
                submit_parts "rvar_${METHOD}_n${ns}" \
                    "result/robust/var/${METHOD}/n${ns}" \
                    "--n_samples $ns --estimate_var"
            done;;
        noise)
            for fam in $FAMILIES; do
                for w in $DISTANCES; do
                    # ジョブ名に '.' は使えない場合があるため距離からは除く (0.01 -> w001)
                    submit_parts "rnoise_${METHOD}_${fam}_w${w//./}" \
                        "result/robust/noise/${METHOD}/${fam}_w${w}" \
                        "--n_samples $NOISE_NSAMPLES --noise $fam --distance $w"
                done
            done;;
    esac
done

echo "投入完了: ${count} ジョブ -> result/robust/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/synthetic/plot_robust.py --csv result/robust --n_pvalues 1000"
