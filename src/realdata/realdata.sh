#!/bin/bash
# 実データ実験ランチャ兼ジョブ本体。
#   - 引数なし/オプション付きで実行 -> 全作業を「1本のジョブ配列」で qsub
#   - PBS 配列 subjob (PBS_ARRAY_INDEX つき) -> python を 1 作業単位だけ走らせる
# UCI の 8 データセットを n 行サブサンプリングして p 値を計算し、
# 棄却率 P(p 値 < α) を (データセット × n) ごとに見る (si4pipeline 論文の実験)。
#
# 注意: データセットの取得にはネットワークが必要なため、ランチャ側 (ログイン
# ノード) で先にダウンロードし、subjob はキャッシュ済み CSV のみ読む。
#
# 同時実行本数は キューの設定 (ユーザーあたり同時実行上限) に合わせて絞る。
#   - 既定: qstat からキュー設定の上限を自動検出し、-J 0-(TOTAL-1)%上限 で投入
#   - 自動検出できなければ %throttle を付けず、キュー設定の上限にそのまま従う
#   - -j で上限を明示指定して上書きも可能
#
# 使い方 (プロジェクトルートで実行):
#   bash src/realdata/realdata.sh                        # 既定で投入
#   bash src/realdata/realdata.sh -m lasso -D "wine_red wine_white" -N "100 150 200"
#   bash src/realdata/realdata.sh -j 20                  # 同時実行を 20 本に明示
#
# オプション: -m 手法  -D "データセットのリスト"  -N "サンプル数のリスト"
#             -n 分割数((データセット,n)ごとの作業単位数)  -p 1作業単位のp値数
#             -w ワーカー数  -q キュー  -j 同時実行上限(空=キュー設定に従う)

#PBS -N realdata
#PBS -j oe

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- ジョブ本体: 配列 subjob (PBS_ARRAY_INDEX つき) として呼ばれた場合 ---
# 配列インデックス k から (データセット ds, サンプル数 ns, 分割 i) を復元する。
#   k = (ds_index * |N| + ns_index) * NSPLIT + i
if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    cd "$PBS_O_WORKDIR"
    # 計算ノードでは uv を使わず、uv sync --extra experiments 済みの
    # venv の python を直接使う (uv run は extras を外して再同期してしまう)
    PYTHON="$PBS_O_WORKDIR/.venv/bin/python3"
    if [ ! -x "$PYTHON" ]; then
        echo "エラー: $PYTHON がありません。先に uv sync --extra experiments を実行してください。" >&2
        exit 1
    fi
    read -ra DS_ARR <<< "${DATASETS//:/ }"          # ':' 符号化を復号
    read -ra NS_ARR <<< "${NLIST//:/ }"
    n_ns=${#NS_ARR[@]}
    ds=${DS_ARR[$(( PBS_ARRAY_INDEX / (NSPLIT * n_ns) ))]}
    ns=${NS_ARR[$(( (PBS_ARRAY_INDEX / NSPLIT) % n_ns ))]}
    i=$(( PBS_ARRAY_INDEX % NSPLIT ))
    s=$(( i * PER )); e=$(( (i + 1) * PER ))
    OUTDIR=result/realdata/${METHOD}/${ds}_n${ns}
    mkdir -p "$OUTDIR"
    "$PYTHON" src/realdata/run_realdata.py \
        --methods "$METHOD" --datasets "$ds" --n_samples "$ns" \
        --seed_start "$s" --seed_end "$e" --seed_offset "$OFFSET" \
        --workers "$WORKERS" --out "$OUTDIR/part_${i}.csv"
    exit 0
fi

# --- ランチャ: 全作業を 1 本のジョブ配列として投入 ---
source "$(dirname "$0")/../pbs_lib.sh"

# PBS ログ (標準出力/エラー) の出力先を用意する
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EO_DIR="$PROJECT_DIR/eo"
mkdir -p "$EO_DIR"

METHOD=lasso
DATASETS="airfoil concrete energy_heating energy_cooling gas_turbine real_estate wine_red wine_white"
NLIST="100 150 200"          # サブサンプルのサイズ n のリスト (論文と同じ)
NSPLIT=10                    # (データセット, n) ごとの作業単位数 (= 分割数)
PER=110                      # 1 作業単位が計算する p 値の個数 (10 分割 × 110 = 1100、エラー時の不足に備え 1 割の余裕)
WORKERS=96
QUEUE=cpu-large
THROTTLE=""                  # -j: 空ならキュー設定から自動検出

while getopts "m:D:N:n:p:w:q:j:" o; do case $o in
    m) METHOD=$OPTARG;; D) DATASETS=$OPTARG;; N) NLIST=$OPTARG;;
    n) NSPLIT=$OPTARG;; p) PER=$OPTARG;; w) WORKERS=$OPTARG;;
    q) QUEUE=$OPTARG;; j) THROTTLE=$OPTARG;;
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
"$PROJECT_DIR/.venv/bin/python3" src/realdata/datasets.py

# 作業単位数 = データセット数 × n の個数 × 分割数
read -ra DS_ARR <<< "$DATASETS"
read -ra NS_ARR <<< "$NLIST"
TOTAL=$(( ${#DS_ARR[@]} * ${#NS_ARR[@]} * NSPLIT ))

# 同時実行スロット数を決定 (-j 指定 > qstat 自動検出 > 空=キュー設定に従う)
LIMIT=$(resolve_throttle "$THROTTLE" "$QUEUE")
if [ -n "$LIMIT" ]; then
    RANGE="0-$((TOTAL - 1))%${LIMIT}"
    echo "同時実行上限 ${LIMIT} 本で ${TOTAL} 作業単位を配列ジョブとして投入します (キュー: ${QUEUE})。"
else
    RANGE="0-$((TOTAL - 1))"
    echo "同時実行上限を自動検出できませんでした。キュー ${QUEUE} の設定に従って ${TOTAL} 作業単位を投入します。"
fi

# 出力先を事前に用意 (多数 subjob からの同時 mkdir 競合を避ける)
for ds in "${DS_ARR[@]}"; do
    for ns in "${NS_ARR[@]}"; do mkdir -p "result/realdata/${METHOD}/${ds}_n${ns}"; done
done

# subjob へ渡す設定 (リストは ':' 符号化して空白問題を回避)
VLIST="METHOD=${METHOD},DATASETS=$(encode_list "$DATASETS"),NLIST=$(encode_list "$NLIST"),NSPLIT=${NSPLIT},PER=${PER},WORKERS=${WORKERS},OFFSET=${OFFSET}"

qsub -q "$QUEUE" -J "$RANGE" -o "$EO_DIR/" \
     -l select=1:ncpus=${WORKERS}:mem=96gb -l walltime=24:00:00 \
     -v "$VLIST" "$0"

echo "投入完了 -> result/realdata/${METHOD}/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/realdata/plot_realdata.py --csv result/realdata"
