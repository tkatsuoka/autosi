#!/bin/bash
# gas_turbine だけを再投入するドライバ。
#
# gas_turbine は 8 データセット中で最も探索が重く、autosi のメモ化キャッシュが
# 無制限に増える問題 (autosi/poly_memo.py の MAX_ENTRIES で修正済み) の影響を
# 最も強く受けていた。他のデータセットが完了済み・走行中の状態で、
# gas_turbine だけを投げ直すために使う。
#
# 投入内容は realdata.sh に委譲するため、分割数・p 値数・コア数・seed の
# 割り当てはすべて他データセットと同一になる (結果の比較可能性を保つ)。
#
# seed ブロックは他データセットと同一 (part 番号 × SEED_STRIDE、既定で part 0〜20)。
# 走行中のジョブと同じ seed = 同じサブサンプルなので、結果は決定的に一致する。
# 同じ part を投げ直した場合は同一内容で上書きされるだけで、集計は変わらない。
#
# 使い方 (プロジェクトルートで実行):
#   bash src/realdata/gas_turbine.sh                       # cv_r2_lasso のみ
#   bash src/realdata/gas_turbine.sh -m "lasso cv_r2_lasso" -q cpu-huge
#   bash src/realdata/gas_turbine.sh -M -t none            # 未完了 part のみ・時間上限なしで再投入
#
# オプション: -m "手法のリスト" (既定: cv_r2_lasso)  -q キュー
#             -o part番号の開始  -n 分割数  -p 1ジョブのp値数  -t 実行時間上限
#             -M 未完了 (CSV が無い) part のみ投入
#             (-o / -n / -p / -t / -M は realdata.sh にそのまま渡す。-o を変えると
#              seed ブロックも変わるので、通常は既定のままにすること)

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

METHODS="cv_r2_lasso"        # 対象手法 (空白区切り)。重いのは cv_r2_lasso のみ
PASS_OPTS=()                 # realdata.sh へそのまま渡すオプション

while getopts "m:q:o:n:p:t:M" o; do case $o in
    m) METHODS=$OPTARG;;
    q) PASS_OPTS+=(-q "$OPTARG");;
    o) PASS_OPTS+=(-o "$OPTARG");;
    n) PASS_OPTS+=(-n "$OPTARG");;
    p) PASS_OPTS+=(-p "$OPTARG");;
    t) PASS_OPTS+=(-t "$OPTARG");;
    M) PASS_OPTS+=(-M);;
esac; done

for m in $METHODS; do
    echo "=== $m / gas_turbine を投入します ==="
    bash "$DIR/realdata.sh" -m "$m" -D gas_turbine \
        ${PASS_OPTS[@]+"${PASS_OPTS[@]}"}
done

echo "完了後の確認:"
echo "  find result/realdata -path '*gas_turbine*' -name 'part_*.csv' | wc -l"
echo "  uv run --extra experiments python src/realdata/plot_realdata.py --csv result/realdata --n_pvalues 100"
