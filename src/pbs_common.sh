#!/bin/bash
# =============================================================
#  実験投入スクリプト共通の設定・ヘルパ
#  - 変更が必要なパラメータは基本ここに集約する
#  - ドライバ (synthetic/*.sh, realdata/*.sh) から `source` して使う想定
#    （PBS ジョブ本体 jobs/*.pbs は spool 実行のため source しない）
# =============================================================

# --- 実行環境 ------------------------------------------------
#  プロジェクトの場所はこのファイルの位置から自動導出する
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # <project>/src
PROJECT_DIR="$(cd "${SRC_DIR}/.." && pwd)"                    # <project>
#  Python は uv 管理の仮想環境を使う（プロジェクト直下で
#  `uv sync --extra experiments` して作る。jobs/*.pbs も同じ .venv を
#  PBS_O_WORKDIR からの相対パスで参照する）
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
EO_DIR="${PROJECT_DIR}/eo"          # PBS 標準出力/エラーの出力先
mkdir -p "$EO_DIR"                  # 無ければ作成（投入前に用意しておく）

# --- PBS 投入設定 --------------------------------------------
#  リソース要求はここで一元管理し、submit_job が qsub コマンドラインで渡す
#  （コマンドライン指定は jobs/*.pbs 内の #PBS ディレクティブより優先される）
QUEUE="cpu-large"                   # 既定キュー。ドライバの -q で cpu-huge に切替可能
WORKERS=96                          # --workers（select の ncpus にもこの値を使う）
MEM_PER_NODE="512gb"                # 1 ノードあたりのメモリ要求（キューによらず共通）
WALLTIME="72:00:00"                 # 実行時間上限（キューによらず共通）。
                                    # "none" にすると qsub に walltime を渡さず、
                                    # キューの既定値/上限に従う（キューに上限が
                                    # 無ければ時間無制限になる。qstat -Qf で確認）

# --- ワーカースクリプト（qsub で投入する実体） ---------------
SYNTHETIC_JOB="${SRC_DIR}/jobs/synthetic.pbs"   # src/synthetic/run_experiment.py 用
REALDATA_JOB="${SRC_DIR}/jobs/realdata.pbs"     # src/realdata/run_realdata.py 用

# --- 投入ヘルパ ----------------------------------------------
#  出力先(-o/-e)は EO_DIR を明示指定する（jobs/*.pbs 側には固定パスを書かない）
#  1 ジョブ投入する
#    使い方: submit_job <ジョブ名> <jobスクリプト> "<MYARGS 文字列>"
submit_job() {
    local job_name="$1"
    local job_script="$2"
    local myargs="$3"
    # WALLTIME="none" のときは walltime 要求を付けない（キューの既定に従う）
    local wt_opts=()
    if [ -n "$WALLTIME" ] && [ "$WALLTIME" != "none" ]; then
        wt_opts=(-l "walltime=${WALLTIME}")
    fi
    qsub -N "$job_name" -q "$QUEUE" \
         -l "select=1:ncpus=${WORKERS}:mem=${MEM_PER_NODE}" \
         ${wt_opts[@]+"${wt_opts[@]}"} \
         -o "$EO_DIR" -e "$EO_DIR" \
         -v MYARGS="$myargs" "$job_script"
}
