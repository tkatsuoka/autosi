#!/bin/bash
# 合成データ実験の一括投入ドライバ。
# 3 手法 (marginal_screening / lasso / cv_r2_lasso) × データ種別 (iid / corr) の
# 全組み合わせで FPR / power 実験をまとめて qsub する。
# ロバスト性実験 (robust) は iid のみなので 3 手法だけをループする (明示指定時のみ)。
# 個別の条件だけ投入したい場合は fpr.sh / power.sh / robust.sh を直接使う。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/submit_all.sh                    # fpr + power 両方 (480 ジョブ)
#   bash src/synthetic/submit_all.sh fpr                # FPR のみ (240 ジョブ)
#   bash src/synthetic/submit_all.sh power              # power のみ (240 ジョブ)
#   bash src/synthetic/submit_all.sh robust             # ロバスト性のみ (720 ジョブ)
#   bash src/synthetic/submit_all.sh fpr -q cpu-huge    # オプションは各ドライバへ転送
#
# 実験名 (fpr / power / robust) の後ろに書いたオプション ("-" 始まり以降すべて) は
# 各ドライバにそのまま渡される (-q キュー、-n 分割数 など)。

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

METHODS="marginal_screening lasso cv_r2_lasso"
DATA_TYPES="iid corr"

# 先頭の実験名の並びと、最初のオプション以降 (ドライバへの転送分) に分ける
EXPERIMENTS=""
PASS_OPTS=()
while [ "$#" -gt 0 ]; do
    case $1 in
        fpr|power|robust) EXPERIMENTS+="$1 "; shift;;
        -*)               PASS_OPTS=("$@"); break;;
        *) echo "エラー: 未知の実験 '$1' (指定可能: fpr / power / robust)" >&2; exit 1;;
    esac
done
# 実験名の指定がなければ fpr と power の両方を投入する
[ -n "$EXPERIMENTS" ] || EXPERIMENTS="fpr power"

for exp in $EXPERIMENTS; do
    for m in $METHODS; do
        # ロバスト性実験は iid 固定なのでデータ種別のループを回さない
        if [ "$exp" = robust ]; then
            echo "=== robust : ${m} ==="
            bash "$DIR/robust.sh" -m "$m" ${PASS_OPTS[@]+"${PASS_OPTS[@]}"}
            continue
        fi
        for d in $DATA_TYPES; do
            echo "=== ${exp} : ${m} / ${d} ==="
            # PASS_OPTS が空でも set -u で落ちないよう安全に展開する
            bash "$DIR/${exp}.sh" -m "$m" -d "$d" ${PASS_OPTS[@]+"${PASS_OPTS[@]}"}
        done
    done
done
