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
QUEUE="cpu-large"                   # キューは全実験で cpu-large に統一（jobs/*.pbs にも記載）
WORKERS=96                          # --workers（jobs/*.pbs の ncpus と揃える）

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
    qsub -N "$job_name" -q "$QUEUE" -o "$EO_DIR" -e "$EO_DIR" \
         -v MYARGS="$myargs" "$job_script"
}
