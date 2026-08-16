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
#             -o part番号の開始 (不足条件の追い炊き投入時に既存 part の
#                上書きと seed の重複を避ける。例: -o 21 -n 1 -p 5)
#             -t 実行時間上限 (既定 240:00:00。キューの上限以内で指定すること。
#                上限に達したジョブは CSV を書かずに失われるため、計算が重い
#                条件では長めに取る。"none" で walltime 要求を付けずに投入し、
#                キューの既定値/上限に従う)
#             -M 未完了の part のみ投入 (出力先に part_i.csv が既にあるものは
#                完了済みとみなしてスキップ。walltime 超過などで CSV を書けずに
#                終わった part だけを同じ seed で投げ直す)
#             -w ワーカー数  -q キュー

set -euo pipefail
source "$(dirname "$0")/../pbs_common.sh"

# 実データは 1 条件 105 試行と小さいため、(条件 × 21 分割) の小さなジョブに
# 割って並列度を稼ぐ: 1 ジョブ = 5 試行 ÷ 5 コア = 1 巡で完了する。
# 粒度を細かくするほどユーザーあたりのコア数上限の枠に隙間なく詰められる。
# メモリは 96 コアノードに 19 ジョブ相乗りできるよう 512gb ÷ 19 ≈ 26gb に按分。
# cv_r2_lasso でも余裕を持てるよう実行時間上限は 240 時間にする (-w で変更可)。
WORKERS=5
MEM_PER_NODE="26gb"
WALLTIME="240:00:00"

METHOD=lasso
DATASETS="airfoil concrete energy_heating energy_cooling gas_turbine real_estate wine_red wine_white"
NLIST="100"                  # サブサンプルのサイズ n のリスト (-N "100 150 200" で論文と同じ 3 水準に)
NSPLIT=21                    # (データセット, n) ごとのジョブ数 (= 分割数)
PER=5                        # 1 ジョブが集める p 値の本数 (21 分割 × 5 = 105 本/条件。
                             # 動的モードで失敗分 (空の特徴集合など) は自動追加され
                             # 本数が保証されるため、余裕は目標 100 本 + 5 本で足りる)
SEED_STRIDE=1000             # part ごとの seed ブロック幅 (動的モードは試行数が変動するため重複防止。
                             # 1 part が消費する seed は最大 max_trials = 50 × PER = 250 個)
ISTART=0                     # part 番号の開始 (-o。追い炊き投入時に既存 part と衝突させない)
ONLY_MISSING=0               # -M: 既に CSV がある part をスキップ (未完了分の再投入)

while getopts "m:D:N:n:p:o:t:w:q:M" o; do case $o in
    m) METHOD=$OPTARG;; D) DATASETS=$OPTARG;; N) NLIST=$OPTARG;;
    n) NSPLIT=$OPTARG;; p) PER=$OPTARG;; o) ISTART=$OPTARG;;
    t) WALLTIME=$OPTARG;; w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;;
    M) ONLY_MISSING=1;;
esac; done

# 手法ごとに seed をずらし、手法間で異なるサブサンプル (独立) にする。
# ブロック幅 100000 は 1 条件が使う seed 幅 (21 part × SEED_STRIDE ≈ 21000) より
# 十分大きく取ってある。
case $METHOD in
    marginal_screening) OFFSET=0;;
    lasso)              OFFSET=100000;;
    cv_r2_lasso)        OFFSET=200000;;
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
        for i in $(seq $ISTART $((ISTART + NSPLIT - 1))); do
            # -M: CSV が既にある part は完了済み (walltime 超過ジョブは CSV を
            #     書かない) なのでスキップし、欠けている part だけを投げ直す
            if [ "$ONLY_MISSING" = 1 ] && [ -f "$OUTDIR/part_${i}.csv" ]; then
                echo "  スキップ: part_${i} (完了済み)"
                continue
            fi
            # 動的モード: 各 part は成功した p 値 PER 本を集めるまで試行を追加する。
            # 消費する seed 数が変動するため、part ごとに SEED_STRIDE 刻みの
            # ブロックを割り当てて重複を防ぐ
            MYARGS="--methods $METHOD --datasets $ds --n_samples $ns"
            MYARGS+=" --target_pvalues $PER --seed_start $((i * SEED_STRIDE))"
            MYARGS+=" --seed_offset $OFFSET"
            MYARGS+=" --workers $WORKERS --out $OUTDIR/part_${i}.csv"
            submit_job "rd_${METHOD}_${ds}_n${ns}_${i}" "$REALDATA_JOB" "$MYARGS"
            count=$((count + 1))
        done
    done
done

echo "投入完了: ${count} ジョブ -> result/realdata/${METHOD}/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/realdata/plot_realdata.py --csv result/realdata"
