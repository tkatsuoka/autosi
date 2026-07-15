#!/bin/bash
# 検出力 (power) 実験ランチャ兼ジョブ本体。
#   - 引数なし/オプション付きで実行 -> 全作業を「1本のジョブ配列」で qsub
#   - PBS 配列 subjob (PBS_ARRAY_INDEX つき) -> python を 1 作業単位だけ走らせる
# y = Xβ + ε (先頭 N_SIGNAL 個の係数を Δ) でデータを生成し、
# power = P(p 値 < α) を Δ ごとに見る。
#
# 同時実行本数はキューの設定 (ユーザーあたり同時実行上限) に合わせて絞る。
#   - 既定: qstat からキュー設定の上限を自動検出し、-J 0-(TOTAL-1)%上限 で投入
#   - 自動検出できなければ %throttle を付けず、キュー設定の上限にそのまま従う
#   - -j で上限を明示指定して上書きも可能
#
# 使い方 (プロジェクトルートで実行):
#   bash src/synthetic/power.sh                                  # 既定で投入
#   bash src/synthetic/power.sh -m lasso -d iid -n 10 -p 1000 -s "0.1 0.2 0.3 0.4"
#   bash src/synthetic/power.sh -j 20                            # 同時実行を 20 本に明示
#
# オプション: -m 手法  -d データ種別  -n 分割数(Δごとの作業単位数)
#             -p 1作業単位のp値数  -s "Δのリスト"  -k 信号特徴量数
#             -w ワーカー数  -q キュー  -j 同時実行上限(空=キュー設定に従う)

#PBS -N synthetic_power
#PBS -j oe

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- ジョブ本体: 配列 subjob (PBS_ARRAY_INDEX つき) として呼ばれた場合 ---
# 配列インデックス k から (信号 sig, 分割 i) を復元して 1 作業単位を実行する。
#   k = sig_index * NSPLIT + i
if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    cd "$PBS_O_WORKDIR"
    # 計算ノードでは uv を使わず、uv sync --extra experiments 済みの
    # venv の python を直接使う (uv run は extras を外して再同期してしまう)
    PYTHON="$PBS_O_WORKDIR/.venv/bin/python3"
    if [ ! -x "$PYTHON" ]; then
        echo "エラー: $PYTHON がありません。先に uv sync --extra experiments を実行してください。" >&2
        exit 1
    fi
    read -ra SIG_ARR <<< "${SIGNALS//:/ }"          # ':' 符号化を復号
    sig=${SIG_ARR[$(( PBS_ARRAY_INDEX / NSPLIT ))]}
    i=$(( PBS_ARRAY_INDEX % NSPLIT ))
    s=$(( i * PER )); e=$(( (i + 1) * PER ))
    OUTDIR=result/power/${METHOD}_${DATA_TYPE}_d${sig}
    mkdir -p "$OUTDIR"
    "$PYTHON" src/synthetic/run_experiment.py \
        --methods "$METHOD" --data_types "$DATA_TYPE" \
        --signal "$sig" --n_signal "$N_SIGNAL" \
        --seed_start "$s" --seed_end "$e" --seed_offset "$OFFSET" --workers "$WORKERS" \
        --out "$OUTDIR/part_${i}.csv"
    exit 0
fi

# --- ランチャ: 全作業を 1 本のジョブ配列として投入 ---
source "$(dirname "$0")/../pbs_lib.sh"

# PBS ログ (標準出力/エラー) の出力先を用意する
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EO_DIR="$PROJECT_DIR/eo"
mkdir -p "$EO_DIR"

METHOD=lasso
DATA_TYPE=iid
NSPLIT=10                    # Δ ごとの作業単位数 (= 分割数)
PER=1100                     # 1 作業単位が計算する p 値の個数 (エラー時の不足に備え 1 割の余裕)
SIGNALS="0.1 0.2 0.3 0.4"    # 横軸となる信号 Δ のリスト
N_SIGNAL=5
WORKERS=96
QUEUE=cpu-large
THROTTLE=""                  # -j: 空ならキュー設定から自動検出

while getopts "m:d:n:p:s:k:w:q:j:" o; do case $o in
    m) METHOD=$OPTARG;; d) DATA_TYPE=$OPTARG;; n) NSPLIT=$OPTARG;;
    p) PER=$OPTARG;; s) SIGNALS=$OPTARG;; k) N_SIGNAL=$OPTARG;;
    w) WORKERS=$OPTARG;; q) QUEUE=$OPTARG;; j) THROTTLE=$OPTARG;;
esac; done

# 手法ごとに seed をずらし、手法間で異なるデータ (独立) にする
case $METHOD in
    marginal_screening) OFFSET=0;;
    lasso)              OFFSET=1000000;;
    cv_r2_lasso)        OFFSET=2000000;;
    *)                  OFFSET=0;;
esac

# 作業単位数 = Δ の個数 × 分割数
read -ra SIG_ARR <<< "$SIGNALS"
TOTAL=$(( ${#SIG_ARR[@]} * NSPLIT ))

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
for sig in "${SIG_ARR[@]}"; do mkdir -p "result/power/${METHOD}_${DATA_TYPE}_d${sig}"; done

# subjob へ渡す設定 (リストは ':' 符号化して空白問題を回避)
VLIST="METHOD=${METHOD},DATA_TYPE=${DATA_TYPE},SIGNALS=$(encode_list "$SIGNALS"),NSPLIT=${NSPLIT},PER=${PER},N_SIGNAL=${N_SIGNAL},WORKERS=${WORKERS},OFFSET=${OFFSET}"

qsub -q "$QUEUE" -J "$RANGE" -o "$EO_DIR/" \
     -l select=1:ncpus=${WORKERS}:mem=256gb -l walltime=72:00:00 \
     -v "$VLIST" "$0"

echo "投入完了 -> result/power/ 以下 (PBS ログは eo/)。完了後:"
echo "  uv run --extra experiments python src/synthetic/plot_power.py --csv result/power"
