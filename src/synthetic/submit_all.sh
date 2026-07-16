#!/bin/bash
# 合成データ実験の一括投入ドライバ。
# 3 手法 (marginal_screening / lasso / cv_r2_lasso) × データ種別 (iid / corr) の
# 全組み合わせで FPR / power 実験をまとめて qsub する。
# 個別の条件だけ投入したい場合は fpr.sh / power.sh を直接使う。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/submit_all.sh          # fpr + power 両方 (480 ジョブ)
#   bash src/synthetic/submit_all.sh fpr      # FPR のみ (240 ジョブ)
#   bash src/synthetic/submit_all.sh power    # power のみ (240 ジョブ)

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

METHODS="marginal_screening lasso cv_r2_lasso"
DATA_TYPES="iid corr"

# 引数なしなら fpr と power の両方を投入する
if [ "$#" -eq 0 ]; then
    EXPERIMENTS="fpr power"
else
    EXPERIMENTS="$*"
fi
for exp in $EXPERIMENTS; do
    case $exp in
        fpr|power) ;;
        *) echo "エラー: 未知の実験 '$exp' (指定可能: fpr / power)" >&2; exit 1;;
    esac
done

for exp in $EXPERIMENTS; do
    for m in $METHODS; do
        for d in $DATA_TYPES; do
            echo "=== ${exp} : ${m} / ${d} ==="
            bash "$DIR/${exp}.sh" -m "$m" -d "$d"
        done
    done
done
